import base64
import json
import struct
import sys
import zlib
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from dify_plugin.entities.tool import ToolConfiguration, ToolInvokeMessage, ToolProviderConfiguration
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from openai import AzureOpenAI, OpenAI

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from utils.model_capabilities import validate_image_parameters


@pytest.mark.parametrize(
    "messages, succeeds",
    [
        (["image"], True),
        (["text", "image"], True),
        (["text", "text"], False),
        ([], False),
        (["empty_image"], False),
        (["non_image"], False),
        (["missing_mime"], False),
    ],
)
def test_provider_validates_generated_image(monkeypatch, messages, succeeds):
    from provider import azure_openai_tool

    def blob_message(blob=b"image", mime_type="image/png"):
        return ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.BLOB,
            message=ToolInvokeMessage.BlobMessage(blob=blob),
            meta={"mime_type": mime_type} if mime_type is not None else None,
        )

    available_messages = {
        "image": blob_message(),
        "text": ToolInvokeMessage(
            type=ToolInvokeMessage.MessageType.TEXT,
            message=ToolInvokeMessage.TextMessage(text="Test tool response"),
        ),
        "empty_image": blob_message(blob=b""),
        "non_image": blob_message(blob=b"test document", mime_type="application/pdf"),
        "missing_mime": blob_message(mime_type=None),
    }
    tool = Mock()
    tool.invoke.return_value = iter(available_messages[name] for name in messages)
    factory = Mock(return_value=tool)
    monkeypatch.setattr(azure_openai_tool.ImageGenerateTool, "from_credentials", factory)
    credentials = {"azure_openai_api_key": "test-key"}

    if succeeds:
        azure_openai_tool.AzureOpenAIProvider._validate_credentials(SimpleNamespace(), credentials)
    else:
        with pytest.raises(ToolProviderCredentialValidationError, match="no image was returned") as error:
            azure_openai_tool.AzureOpenAIProvider._validate_credentials(SimpleNamespace(), credentials)
        assert isinstance(error.value.__cause__, ValueError)
        if "text" in messages:
            assert "Tool response: Test tool response\nTest tool response" in str(error.value)

    factory.assert_called_once_with(credentials, user_id="")
    tool.invoke.assert_called_once_with(
        tool_parameters={"prompt": "A plain white square.", "size": "1024x1024", "quality": "low", "n": 1}
    )


def test_provider_preserves_generation_exception(monkeypatch):
    from provider import azure_openai_tool

    tool = Mock()
    original_error = RuntimeError("Test generation failure")
    tool.invoke.side_effect = original_error
    monkeypatch.setattr(azure_openai_tool.ImageGenerateTool, "from_credentials", Mock(return_value=tool))

    with pytest.raises(ToolProviderCredentialValidationError, match="Test generation failure") as error:
        azure_openai_tool.AzureOpenAIProvider._validate_credentials(SimpleNamespace(), {})
    assert error.value.__cause__ is original_error


@pytest.mark.parametrize("api_version", [None, "", "2025-04-01-preview"])
@pytest.mark.parametrize("suffix", ["/openai/v1", "/openai/v1/", "/gateway/openai/v1/"])
def test_create_image_client_uses_v1_endpoint(monkeypatch, api_version, suffix):
    from utils import azure_client

    v1_client = Mock()
    dated_client = Mock()
    monkeypatch.setattr(azure_client, "OpenAI", v1_client)
    monkeypatch.setattr(azure_client, "AzureOpenAI", dated_client)
    credentials = {
        "azure_openai_api_key": "test-key",
        "azure_openai_base_url": "https://example.openai.azure.com" + suffix,
        "azure_openai_api_model_name": "arbitrary-deployment",
    }
    if api_version is not None:
        credentials["azure_openai_api_version"] = api_version

    assert azure_client.create_image_client(credentials) is v1_client.return_value
    v1_client.assert_called_once_with(
        api_key="test-key", base_url=credentials["azure_openai_base_url"].rstrip("/") + "/",
        default_query={"api-version": "preview"},
    )
    dated_client.assert_not_called()


@pytest.mark.parametrize("api_version", ["2024-02-15-preview", "2025-04-01-preview", "2025-04-01"])
def test_create_image_client_preserves_dated_credentials(monkeypatch, api_version):
    from utils import azure_client

    v1_client = Mock()
    dated_client = Mock()
    monkeypatch.setattr(azure_client, "OpenAI", v1_client)
    monkeypatch.setattr(azure_client, "AzureOpenAI", dated_client)
    credentials = {
        "azure_openai_api_key": "test-key",
        "azure_openai_base_url": "https://example.openai.azure.com/",
        "azure_openai_api_model_name": "arbitrary-deployment",
        "azure_openai_api_version": api_version,
    }

    assert azure_client.create_image_client(credentials) is dated_client.return_value
    dated_client.assert_called_once_with(
        api_key="test-key", azure_endpoint=credentials["azure_openai_base_url"],
        api_version=api_version, azure_deployment="arbitrary-deployment",
    )
    v1_client.assert_not_called()


@pytest.mark.parametrize("api_version", [None, "", " "])
def test_create_image_client_does_not_switch_to_v1_on_empty_version(monkeypatch, api_version):
    from utils import azure_client

    v1_client = Mock()
    monkeypatch.setattr(azure_client, "OpenAI", v1_client)
    credentials = {
        "azure_openai_api_key": "test-key",
        "azure_openai_base_url": "https://example.openai.azure.com/",
        "azure_openai_api_model_name": "arbitrary-deployment",
    }
    if api_version is not None:
        credentials["azure_openai_api_version"] = api_version

    with pytest.raises(ValueError, match="API Version is required"):
        azure_client.create_image_client(credentials)
    v1_client.assert_not_called()


def test_provider_requires_no_model_profile(monkeypatch):
    monkeypatch.chdir(PLUGIN_ROOT)
    provider_path = PLUGIN_ROOT / "provider" / "azure_openai_tool.yaml"
    with provider_path.open(encoding="utf-8") as provider_file:
        provider_data = yaml.safe_load(provider_file)
    provider = ToolProviderConfiguration.model_validate(provider_data)
    assert {item.name for item in provider.credentials_schema} == {
        "azure_openai_api_key", "azure_openai_api_model_name",
        "azure_openai_base_url", "azure_openai_api_version",
    }
    api_version = next(item for item in provider.credentials_schema if item.name == "azure_openai_api_version")
    assert api_version.required is False
    assert api_version.default is None


def test_image_parameter_defaults_are_unchanged():
    assert validate_image_parameters({}) == {
        "size": "1024x1024", "quality": "high"
    }


@pytest.mark.parametrize("quality", ["auto", "low", "medium", "high", "xhigh", "max"])
@pytest.mark.parametrize("size", ["1024x1024", "1536x1024", "1024x1536", "auto"])
def test_common_image_options_are_accepted(quality, size):
    assert validate_image_parameters({"size": size, "quality": quality}) == {
        "size": size, "quality": quality
    }


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        ({"quality": "ultra"}, "Invalid quality"),
        ({"quality": []}, "Invalid quality"),
        ({"quality": None}, "Invalid quality"),
        ({"size": "small"}, "Invalid size"),
        ({"size": []}, "Invalid size"),
        ({"size": None}, "Invalid size"),
        ({"size": "custom"}, "Invalid custom_size"),
        ({"size": "custom", "custom_size": "1024X1024"}, "Invalid custom_size"),
        ({"size": "custom", "custom_size": "1024x1024\n"}, "Invalid custom_size"),
        ({"size": "custom", "custom_size": []}, "Invalid custom_size"),
    ],
)
def test_invalid_image_parameters_are_rejected(parameters, error):
    with pytest.raises(ValueError, match=error):
        validate_image_parameters(parameters)


@pytest.mark.parametrize("size", [
    "123x456", "1024x640", "3840x2160", "0x1024", "1025x1024",
    "3856x2160", "3088x1024", "1024x624", "3840x2176",
])
def test_custom_size_constraints_are_left_to_api(size):
    assert validate_image_parameters({"size": "custom", "custom_size": size})["size"] == size


@pytest.fixture(params=["generate", "edit"])
def image_tool(request, monkeypatch):
    from tools import image_edit, image_generate
    from utils import azure_client

    module = image_generate if request.param == "generate" else image_edit
    tool_class = module.ImageGenerateTool if request.param == "generate" else module.ImageEditTool
    api_call = Mock(return_value=SimpleNamespace(data=[]))
    client = SimpleNamespace(images=SimpleNamespace(**{request.param: api_call}))
    monkeypatch.setattr(azure_client, "AzureOpenAI", Mock(return_value=client))
    monkeypatch.setattr(azure_client, "OpenAI", Mock(return_value=client))
    credentials = {
        "azure_openai_api_key": "test-key",
        "azure_openai_base_url": "https://example.openai.azure.com/",
        "azure_openai_api_version": "2025-04-01-preview",
        "azure_openai_api_model_name": "arbitrary-deployment",
    }
    tool = SimpleNamespace(
        runtime=SimpleNamespace(credentials=credentials),
        create_text_message=lambda text: text,
        create_blob_message=lambda blob, meta: {"blob": blob, "meta": meta},
        _decode_image=tool_class._decode_image,
    )
    parameters = {"prompt": "A test image"}
    if request.param == "edit":
        class InputFile:
            filename = "input.png"
            blob = b"test image"

        monkeypatch.setattr(module, "File", InputFile)
        parameters["image"] = InputFile()

    return tool_class, tool, parameters, api_call


@pytest.fixture
def v1_image_tool(image_tool):
    image_tool[1].runtime.credentials["azure_openai_base_url"] = "https://example.openai.azure.com/openai/v1/"
    image_tool[1].runtime.credentials.pop("azure_openai_api_version")
    return image_tool






@pytest.mark.parametrize("api_version", ["2024-02-15-preview", "2025-04-01-preview", "2025-04-01"])
@pytest.mark.parametrize("output_parameters", [{}, {"output_format": "jpeg", "output_compression": 50}, {"output_format": "png", "output_compression": -1}])
def test_tools_preserve_dated_request_parameters(image_tool, api_version, output_parameters):
    tool_class, tool, parameters, api_call = image_tool
    tool.runtime.credentials["azure_openai_api_version"] = api_version

    assert list(tool_class._invoke(tool, {**parameters, **output_parameters})) == []
    expected = {"prompt": parameters["prompt"], "size": "1024x1024", "quality": "high", "n": 1}
    if "image" in parameters:
        expected["image"] = api_call.call_args.kwargs["image"]
    else:
        expected["output_compression"] = output_parameters.get("output_compression", 100)
    api_call.assert_called_once_with(**expected)


@pytest.mark.parametrize("api_version", [None, "", " "])
def test_tools_report_missing_dated_api_version(image_tool, api_version):
    tool_class, tool, parameters, api_call = image_tool
    if api_version is None:
        tool.runtime.credentials.pop("azure_openai_api_version")
    else:
        tool.runtime.credentials["azure_openai_api_version"] = api_version
    messages = list(tool_class._invoke(tool, parameters))
    assert len(messages) == 1
    assert "API Version is required" in messages[0]
    api_call.assert_not_called()


@pytest.mark.parametrize(
    ("parameters", "expected_size", "expected_quality"),
    [
        ({}, "1024x1024", "high"),
        ({"size": "custom", "custom_size": "123x456"}, "123x456", "high"),
        ({"size": "auto", "quality": "auto"}, "auto", "auto"),
        ({"size": "custom", "custom_size": "3840x2160"}, "3840x2160", "high"),
        ({"size": "auto", "quality": "xhigh"}, "auto", "xhigh"),
        ({"quality": "max"}, "1024x1024", "max"),
        ({"size": "custom", "custom_size": "2560x1440", "quality": "max"}, "2560x1440", "max"),
        ({"size": "custom", "custom_size": "1025x1024"}, "1025x1024", "high"),
    ],
)
@pytest.mark.parametrize("use_v1", [False, True])
def test_tools_send_common_image_parameters(image_tool, parameters, expected_size, expected_quality, use_v1):
    tool_class, tool, defaults, api_call = image_tool
    if use_v1:
        tool.runtime.credentials["azure_openai_base_url"] += "openai/v1/"
        tool.runtime.credentials.pop("azure_openai_api_version")
    list(tool_class._invoke(tool, {**defaults, **parameters}))
    api_call.assert_called_once()
    assert api_call.call_args.kwargs["size"] == expected_size
    assert api_call.call_args.kwargs["quality"] == expected_quality
    assert api_call.call_args.kwargs["prompt"] == defaults["prompt"]
    assert api_call.call_args.kwargs["n"] == 1


@pytest.mark.parametrize(
    ("parameters", "error"),
    [
        ({"quality": "ultra"}, "Invalid quality"),
        ({"quality": []}, "Invalid quality"),
        ({"size": "small"}, "Invalid size"),
        ({"size": []}, "Invalid size"),
        ({"size": "custom"}, "Invalid custom_size"),
        ({"size": "custom", "custom_size": "1024X1024"}, "Invalid custom_size"),
    ],
)
def test_tools_reject_malformed_parameters_before_api_call(image_tool, parameters, error):
    tool_class, tool, defaults, api_call = image_tool
    messages = list(tool_class._invoke(tool, {**defaults, **parameters}))
    assert len(messages) == 1
    assert error in messages[0]
    api_call.assert_not_called()


@pytest.mark.parametrize("previous_key", ["base_model", "validation_profile", "model_profile"])
def test_old_profile_credentials_do_not_restrict_image_parameters(image_tool, previous_key):
    tool_class, tool, parameters, api_call = image_tool
    tool.runtime.credentials[previous_key] = "unknown"
    list(tool_class._invoke(tool, {**parameters, "size": "auto", "quality": "max"}))
    api_call.assert_called_once()
    assert api_call.call_args.kwargs["size"] == "auto"
    assert api_call.call_args.kwargs["quality"] == "max"


@pytest.mark.parametrize("use_v1", [False, True])
def test_tools_report_api_parameter_rejection(image_tool, use_v1):
    tool_class, tool, parameters, api_call = image_tool
    if use_v1:
        tool.runtime.credentials["azure_openai_base_url"] += "openai/v1/"
        tool.runtime.credentials.pop("azure_openai_api_version")
    api_call.side_effect = RuntimeError("This deployment does not support quality max")
    messages = list(tool_class._invoke(tool, {**parameters, "quality": "max"}))
    api_call.assert_called_once()
    assert len(messages) == 1
    assert "This deployment does not support quality max" in messages[0]


@pytest.mark.parametrize("tool_name", ["image_generate", "image_edit"])
def test_tool_options_offer_common_image_parameters(tool_name):
    with (PLUGIN_ROOT / "tools" / f"{tool_name}.yaml").open(encoding="utf-8") as tool_file:
        config = ToolConfiguration.model_validate(yaml.safe_load(tool_file))
    parameters = {parameter.name: parameter for parameter in config.parameters}
    for name, supported_values in [
        ("quality", {"auto", "low", "medium", "high", "xhigh", "max"}),
        ("size", {"1024x1024", "1536x1024", "1024x1536", "auto", "custom"}),
    ]:
        option_values = [option.value for option in parameters[name].options]
        assert len(option_values) == len(set(option_values))
        assert set(option_values) == supported_values
    assert parameters["quality"].default == "high"
    assert parameters["size"].default == "1024x1024"


@pytest.fixture
def encoded_png():
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+ip1sAAAAASUVORK5CYII="
    )


def test_png_fixture_is_valid(encoded_png):
    data = base64.b64decode(encoded_png)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    chunks = {}
    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset:offset + 4], "big")
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = int.from_bytes(data[offset + 8 + length:offset + 12 + length], "big")
        assert zlib.crc32(kind + payload) == crc
        chunks[kind] = payload
        offset += length + 12
    assert offset == len(data)
    assert list(chunks) == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[b"IHDR"]) == (1, 1, 8, 4, 0, 0, 0)
    assert zlib.decompress(chunks[b"IDAT"]) == b"\x01\xff\xff"


@pytest.mark.parametrize("usage_state", ["absent", "null", "details_absent", "details_null", "complete"])
def test_tools_return_image_with_optional_usage(image_tool, usage_state, encoded_png):
    tool_class, tool, parameters, api_call = image_tool
    payload = {"created": 0, "data": [{"b64_json": encoded_png}]}
    expected_usage = {"total_tokens": 30, "input_tokens": 10, "output_tokens": 20}
    if usage_state == "null":
        payload["usage"] = None
    elif usage_state in {"details_absent", "details_null", "complete"}:
        payload["usage"] = dict(expected_usage)
        if usage_state == "details_null":
            payload["usage"]["input_tokens_details"] = None
        elif usage_state == "complete":
            details = {"text_tokens": 4, "image_tokens": 6}
            payload["usage"]["input_tokens_details"] = details
            expected_usage["input_tokens_details"] = details

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with AzureOpenAI(
        api_key="test-key",
        azure_endpoint="https://example.openai.azure.com/",
        api_version="2025-04-01-preview",
        azure_deployment="test-deployment",
        http_client=httpx.Client(transport=transport),
    ) as client:
        api_call.return_value = client.images.generate(model="test-deployment", prompt="test")

    messages = list(tool_class._invoke(tool, parameters))
    api_call.assert_called_once()
    assert len(messages) == 1
    assert messages[0]["blob"] == base64.b64decode(encoded_png)
    assert messages[0]["meta"]["mime_type"] == "image/png"
    usage_key = "token_usage" if tool_class.__name__ == "ImageGenerateTool" else "usage"
    if usage_state in {"absent", "null"}:
        assert usage_key not in messages[0]["meta"]
    else:
        assert messages[0]["meta"][usage_key] == expected_usage


@pytest.mark.parametrize(
    ("output_parameters", "expected_format", "expected_compression"),
    [
        ({}, "png", None),
        ({"output_format": "png", "output_compression": 50}, "png", None),
        ({"output_format": "png", "output_compression": -1}, "png", None),
        ({"output_format": "jpeg"}, "jpeg", 100),
        ({"output_format": "jpeg", "output_compression": 0}, "jpeg", 0),
        ({"output_format": "jpeg", "output_compression": 50.0}, "jpeg", 50),
        ({"output_format": "jpeg", "output_compression": 100}, "jpeg", 100),
    ],
)
def test_tools_send_output_parameters(v1_image_tool, output_parameters, expected_format, expected_compression):
    tool_class, tool, parameters, api_call = v1_image_tool
    assert list(tool_class._invoke(tool, {**parameters, **output_parameters})) == []
    api_call.assert_called_once()
    arguments = api_call.call_args.kwargs
    assert arguments["model"] == "arbitrary-deployment"
    assert arguments["output_format"] == expected_format
    if expected_compression is None:
        assert "output_compression" not in arguments
    else:
        assert arguments["output_compression"] == expected_compression
        assert type(arguments["output_compression"]) is int


@pytest.mark.parametrize(
    ("output_parameters", "error"),
    [
        ({"output_format": "webp"}, "Invalid output_format"),
        ({"output_format": "gif"}, "Invalid output_format"),
        ({"output_format": None}, "Invalid output_format"),
        ({"output_format": []}, "Invalid output_format"),
        *[
            ({"output_format": "jpeg", "output_compression": compression}, "Invalid output_compression")
            for compression in [-1, 101, 50.5, "50", None, True, float("nan"), float("inf")]
        ],
    ],
)
def test_tools_reject_invalid_output_parameters(v1_image_tool, output_parameters, error):
    tool_class, tool, parameters, api_call = v1_image_tool
    messages = list(tool_class._invoke(tool, {**parameters, **output_parameters}))
    assert len(messages) == 1
    assert error in messages[0]
    api_call.assert_not_called()










@pytest.mark.parametrize("prefix", ["", "data:image/png;base64,", "data:image/jpeg;base64,"])
def test_tools_preserve_actual_png_mime_when_jpeg_requested(image_tool, encoded_png, prefix):
    tool_class, tool, parameters, api_call = image_tool
    api_call.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=prefix + encoded_png)])
    messages = list(tool_class._invoke(tool, {**parameters, "output_format": "jpeg"}))
    assert len(messages) == 1
    assert messages[0]["blob"] == base64.b64decode(encoded_png)
    assert messages[0]["meta"]["mime_type"] == "image/png"


@pytest.fixture
def encoded_jpeg():
    return (
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U"
        "HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/"
        "8QAFAABAAAAAAAAAAAAAAAAAAAAB//EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8Af3//2Q=="
    )


@pytest.mark.parametrize("prefix", ["", "data:image/jpeg;base64,", "data:image/png;base64,"])
def test_tools_return_jpeg_with_correct_mime(image_tool, encoded_jpeg, prefix):
    tool_class, tool, parameters, api_call = image_tool
    api_call.return_value = SimpleNamespace(data=[SimpleNamespace(b64_json=prefix + encoded_jpeg)])
    messages = list(tool_class._invoke(tool, {**parameters, "output_format": "jpeg"}))
    assert len(messages) == 1
    assert messages[0]["blob"] == base64.b64decode(encoded_jpeg)
    assert messages[0]["meta"]["mime_type"] == "image/jpeg"


@pytest.mark.parametrize("output_format", ["png", "jpeg"])
@pytest.mark.parametrize("use_v1", [False, True])
def test_tools_serialize_output_parameters_with_sdk(
    image_tool, encoded_png, encoded_jpeg, output_format, use_v1, monkeypatch
):
    from utils import azure_client

    tool_class, tool, parameters, _ = image_tool
    if use_v1:
        tool.runtime.credentials["azure_openai_base_url"] = "https://example.openai.azure.com/openai/v1/"
        tool.runtime.credentials.pop("azure_openai_api_version")
    encoded_image = encoded_png if output_format == "png" else encoded_jpeg
    sent_arguments = {}
    request_urls = []

    def handle_request(request):
        request_urls.append(request.url)
        if use_v1:
            assert request.headers["authorization"] == "Bearer test-key"
        else:
            assert request.headers["api-key"] == "test-key"
        content_type = request.headers["content-type"]
        if content_type.startswith("application/json"):
            sent_arguments.update(json.loads(request.content))
        else:
            headers = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            multipart = BytesParser(policy=policy.default).parsebytes(headers + request.content)
            for part in multipart.iter_parts():
                if part.get_filename() is None:
                    name = part.get_param("name", header="content-disposition")
                    sent_arguments[name] = part.get_payload(decode=True).decode()
        return httpx.Response(200, json={"created": 0, "data": [{"b64_json": encoded_image}], "usage": None})

    if "image" in parameters:
        parameters["image"].blob = base64.b64decode(encoded_png)
    transport = httpx.MockTransport(handle_request)
    with httpx.Client(transport=transport) as http_client:
        monkeypatch.setattr(azure_client, "AzureOpenAI", lambda **kwargs: AzureOpenAI(http_client=http_client, **kwargs))
        monkeypatch.setattr(azure_client, "OpenAI", lambda **kwargs: OpenAI(http_client=http_client, **kwargs))
        messages = list(tool_class._invoke(tool, {
            **parameters, "output_format": output_format, "output_compression": 50,
        }))

    assert len(request_urls) == 1
    expected_action = "edits" if "image" in parameters else "generations"
    if use_v1:
        assert request_urls[0].path == f"/openai/v1/images/{expected_action}"
        assert dict(request_urls[0].params) == {"api-version": "preview"}
        assert sent_arguments["model"] == "arbitrary-deployment"
        assert sent_arguments["output_format"] == output_format
    else:
        assert request_urls[0].path == f"/openai/deployments/arbitrary-deployment/images/{expected_action}"
        assert request_urls[0].params["api-version"] == "2025-04-01-preview"
        assert "model" not in sent_arguments
        assert "output_format" not in sent_arguments
    if (use_v1 and output_format == "jpeg") or (not use_v1 and expected_action == "generations"):
        assert int(sent_arguments["output_compression"]) == 50
    else:
        assert "output_compression" not in sent_arguments
    assert len(messages) == 1
    assert messages[0]["blob"] == base64.b64decode(encoded_image)
    assert messages[0]["meta"]["mime_type"] == f"image/{output_format}"