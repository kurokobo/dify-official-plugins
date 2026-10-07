import base64
import io
from collections.abc import Generator
from typing import Any

from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.file.file import File

from utils.azure_client import create_image_client, is_v1_api_base
from utils.image_output import get_image_mime_type, validate_output_parameters
from utils.image_parameters import validate_image_parameters


class ImageEditTool(Tool):
    """
    Tool to edit images using Azure OpenAI's GPT-image models.
    It takes an input image and a prompt, and optionally a mask,
    to generate an edited version of the image.
    """

    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Invoke the image editing tool.
        """
        use_v1 = is_v1_api_base(self.runtime.credentials["azure_openai_base_url"])
        try:
            client = create_image_client(self.runtime.credentials)
        except ValueError as error:
            yield self.create_text_message(str(error))
            return

        # --- Parameter Extraction and Validation ---
        prompt = tool_parameters.get("prompt")
        if not prompt or not isinstance(prompt, str):
            yield self.create_text_message("Error: Prompt is required.")
            return

        image = tool_parameters.get("image")
        if not image:
            yield self.create_text_message("Error: Input image file is required.")
            return
        edit_args: dict[str, Any] = {
            "prompt": prompt,
        }
        if use_v1:
            edit_args["model"] = self.runtime.credentials["azure_openai_api_model_name"]
        try:
            edit_args.update(validate_image_parameters(tool_parameters))
            edit_args.update(validate_output_parameters(tool_parameters, use_v1=use_v1, is_edit=True))
        except ValueError as error:
            yield self.create_text_message(str(error))
            return

        # Handle single image or array of images
        if isinstance(image, list):
            image_files = []
            for img in image:
                if not isinstance(img, File):
                    yield self.create_text_message("Error: All input images must be valid files.")
                    return
                try:
                    img_bytes = img.blob
                    img_file = io.BytesIO(img_bytes)
                    img_file.name = getattr(img, 'filename', 'input_image.png')
                    image_files.append(img_file)
                except Exception as e:
                    yield self.create_text_message(f"Error processing input image: {e}")
                    # Clean up any opened files
                    for f in image_files:
                        if not f.closed:
                            f.close()
                    return
            edit_args["image"] = image_files
        else:
            if not isinstance(image, File):
                yield self.create_text_message("Error: Input image must be a valid file.")
                return
            try:
                image_bytes = image.blob
                image_file = io.BytesIO(image_bytes)
                image_file.name = getattr(image, 'filename', 'input_image.png')
                edit_args["image"] = image_file
            except Exception as e:
                yield self.create_text_message(f"Error processing input image: {e}")
                return

        # Mask (optional file input)
        mask = tool_parameters.get("mask")
        if mask and isinstance(mask, File):
            try:
                mask_bytes = mask.blob
                mask_file = io.BytesIO(mask_bytes)
                mask_file.name = getattr(mask, 'filename', 'mask_image.png')
                edit_args["mask"] = mask_file
            except Exception as e:
                yield self.create_text_message(f"Warning: Could not process mask image: {e}. Proceeding without mask.")
                if "mask" in edit_args:
                    del edit_args["mask"]

        # Number of images to generate (optional, defaults to 1)
        n = tool_parameters.get("n", 1)
        try:
            n = int(n)
            if not 1 <= n <= 10:
                yield self.create_text_message("Invalid n value. Must be between 1 and 10.")
                return
            edit_args["n"] = n
        except (TypeError, ValueError):
            yield self.create_text_message("Invalid n value. Must be a number between 1 and 10.")
            return

        # --- API Call ---
        try:
            response = client.images.edit(**edit_args)
        except Exception as e:
            # Attempt to close file handles if they exist
            if isinstance(edit_args.get("image"), list):
                for image_file in edit_args["image"]:
                    if not image_file.closed:
                        image_file.close()
            elif "image_file" in locals() and not image_file.closed:
                image_file.close()
            if "mask_file" in locals() and "mask" in edit_args and not mask_file.closed:
                mask_file.close()
            yield self.create_text_message(f"Failed to edit image: {str(e)}")
            return
        finally:
            # Ensure all file handles are closed after API call
            if isinstance(edit_args.get("image"), list):
                for image_file in edit_args["image"]:
                    if not image_file.closed:
                        image_file.close()
            elif "image_file" in locals() and not image_file.closed:
                image_file.close()
            if "mask_file" in locals() and "mask" in edit_args and not mask_file.closed:
                mask_file.close()

        # --- Process Response ---
        has_image = False
        try:
            for image_data in getattr(response, "data", None) or []:
                if not getattr(image_data, "b64_json", None):
                    continue

                try:
                    mime_type, blob_image = self._decode_image(image_data.b64_json)
                    if not blob_image:
                        raise ValueError("Decoded image is empty.")

                    # Create metadata dictionary
                    metadata = {"mime_type": get_image_mime_type(blob_image, mime_type)}

                    # Add usage information if available
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        usage_dict = {}
                        if hasattr(usage, "total_tokens"):
                            usage_dict["total_tokens"] = usage.total_tokens
                        if hasattr(usage, "input_tokens"):
                            usage_dict["input_tokens"] = usage.input_tokens
                        if hasattr(usage, "output_tokens"):
                            usage_dict["output_tokens"] = usage.output_tokens
                        details = getattr(usage, "input_tokens_details", None)
                        if details is not None:
                            usage_dict["input_tokens_details"] = {
                                "text_tokens": details.text_tokens,
                                "image_tokens": details.image_tokens,
                            }
                        if usage_dict:
                            metadata["usage"] = usage_dict

                    yield self.create_blob_message(blob=blob_image, meta=metadata)
                    has_image = True
                except Exception as e:
                    yield self.create_text_message(f"Error processing response image: {str(e)}")
                    continue
        except Exception as e:
            yield self.create_text_message(f"Error processing response: {str(e)}")
            return
        if not has_image:
            yield self.create_text_message("No valid images were returned by the API.")

    @staticmethod
    def _decode_image(base64_image: str) -> tuple[str, bytes]:
        """
        Decode a base64 encoded image. Assumes 'image/png' if no prefix found.
        """
        if ImageEditTool._is_plain_base64(base64_image):
            return "image/png", base64.b64decode(base64_image)
        else:
            return ImageEditTool._extract_mime_and_data(base64_image)

    @staticmethod
    def _is_plain_base64(encoded_str: str) -> bool:
        """Check if the string is plain base64."""
        return not encoded_str.startswith("data:image")

    @staticmethod
    def _extract_mime_and_data(encoded_str: str) -> tuple[str, bytes]:
        """Extract MIME type and data from prefixed base64 string."""
        try:
            mime_type = encoded_str.split(";")[0].split(":")[1]
            image_data_base64 = encoded_str.split(",")[1]
            decoded_data = base64.b64decode(image_data_base64)
            return mime_type, decoded_data
        except Exception:
            # Fallback for potentially malformed strings
            return "image/png", base64.b64decode(encoded_str.split(',')[-1])  # Try decoding last part
