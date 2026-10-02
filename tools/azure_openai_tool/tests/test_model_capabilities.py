import base64
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from dify_plugin.entities.tool import ToolConfiguration, ToolInvokeMessage, ToolProviderConfiguration
from openai import AzureOpenAI, OpenAI

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_ROOT))

from utils.model_capabilities import validate_image_parameters












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
    assert api_version.required is True
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

    module = image_generate if request.param == "generate" else image_edit
    tool_class = module.ImageGenerateTool if request.param == "generate" else module.ImageEditTool
    api_call = Mock(return_value=SimpleNamespace(data=[]))
    client = SimpleNamespace(images=SimpleNamespace(**{request.param: api_call}))
    monkeypatch.setattr(module, "AzureOpenAI", Mock(return_value=client))
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
@pytest.mark.parametrize("use_v1", [False])
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


@pytest.mark.parametrize("use_v1", [False])
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



















