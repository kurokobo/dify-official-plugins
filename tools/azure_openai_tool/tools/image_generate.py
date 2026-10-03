import base64
import random
from collections.abc import Generator
from typing import Any, Dict

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

from utils.azure_client import create_image_client, is_v1_api_base
from utils.image_output import get_image_mime_type, validate_output_parameters
from utils.model_capabilities import validate_image_parameters


class ImageGenerateTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        invoke tools
        """
        use_v1 = is_v1_api_base(self.runtime.credentials["azure_openai_base_url"])
        try:
            client = create_image_client(self.runtime.credentials)
        except ValueError as error:
            yield self.create_text_message(str(error))
            return

        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("Please input prompt")
            return
        # --- Parameter Extraction and Validation ---
        generation_args: Dict[str, Any] = {
            "prompt": prompt,
        }
        if use_v1:
            generation_args["model"] = self.runtime.credentials["azure_openai_api_model_name"]

        try:
            generation_args.update(validate_image_parameters(tool_parameters))
            generation_args.update(validate_output_parameters(tool_parameters, use_v1=use_v1))
        except ValueError as error:
            yield self.create_text_message(str(error))
            return

        # N (optional, defaults to 1)
        n_str = tool_parameters.get("n")
        n = 1  # Default to 1
        if n_str is not None:
            try:
                n = int(n_str)
                if not 1 <= n <= 10:
                    raise ValueError("Number of images (n) must be between 1 and 10.")
            except ValueError as e:
                yield self.create_text_message(f"Invalid n: {e}")
                return
        generation_args["n"] = n  # Include n in generation arguments

        # --- API Call --- 
        try:
            response = client.images.generate(**generation_args)
        except Exception as e:
            yield self.create_text_message(f"Failed to generate image: {str(e)}")
            return

        # --- Process Response --- 
        # Prepare metadata with token usage if available
        metadata = {"mime_type": None}  # Will be set per image
        usage = getattr(response, "usage", None)
        if usage is not None:
            metadata.update({
                "token_usage": {
                    "total_tokens": usage.total_tokens,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens
                }
            })
            details = getattr(usage, "input_tokens_details", None)
            if details is not None:
                metadata["token_usage"]["input_tokens_details"] = {
                    "text_tokens": details.text_tokens,
                    "image_tokens": details.image_tokens
                }

        for image in response.data:
            if not image.b64_json:
                continue
            (mime_type, blob_image) = ImageGenerateTool._decode_image(image.b64_json)
            metadata["mime_type"] = get_image_mime_type(blob_image, mime_type)
            yield self.create_blob_message(blob=blob_image, meta=metadata)

    @staticmethod
    def _decode_image(base64_image: str) -> tuple[str, bytes]:
        """
        Decode a base64 encoded image. If the image is not prefixed with a MIME type,
        it assumes 'image/png' as the default.

        :param base64_image: Base64 encoded image string
        :return: A tuple containing the MIME type and the decoded image bytes
        """
        if ImageGenerateTool._is_plain_base64(base64_image):
            return "image/png", base64.b64decode(base64_image)
        else:
            return ImageGenerateTool._extract_mime_and_data(base64_image)

    @staticmethod
    def _is_plain_base64(encoded_str: str) -> bool:
        """
        Check if the given encoded string is plain base64 without a MIME type prefix.

        :param encoded_str: Base64 encoded image string
        :return: True if the string is plain base64, False otherwise
        """
        return not encoded_str.startswith("data:image")

    @staticmethod
    def _extract_mime_and_data(encoded_str: str) -> tuple[str, bytes]:
        """
        Extract MIME type and image data from a base64 encoded string with a MIME type prefix.

        :param encoded_str: Base64 encoded image string with MIME type prefix
        :return: A tuple containing the MIME type and the decoded image bytes
        """
        try:
            mime_type = encoded_str.split(";")[0].split(":")[1]
            image_data_base64 = encoded_str.split(",")[1]
            decoded_data = base64.b64decode(image_data_base64)
            return mime_type, decoded_data
        except (IndexError, ValueError):
            # Handle potential malformed base64 string gracefully
            # Fallback or raise specific error?
            # Fallback to default png for now
            return "image/png", base64.b64decode(encoded_str)  # Attempt to decode anyway if prefix malformed

    @staticmethod
    def _generate_random_id(length=8):
        characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        random_id = "".join(random.choices(characters, k=length))
        return random_id
