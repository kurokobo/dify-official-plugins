from typing import Any

from dify_plugin import ToolProvider
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

from tools.image_generate import ImageGenerateTool


class AzureOpenAIProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:
            has_image = False
            texts = []
            for result in ImageGenerateTool.from_credentials(credentials, user_id="").invoke(
                tool_parameters={"prompt": "A plain white square.", "size": "1024x1024", "quality": "low", "n": 1}
            ):
                if result.type == ToolInvokeMessage.MessageType.BLOB and isinstance(
                    result.message, ToolInvokeMessage.BlobMessage
                ):
                    mime_type = (result.meta or {}).get("mime_type")
                    if result.message.blob and isinstance(mime_type, str) and mime_type.startswith("image/"):
                        has_image = True
                elif result.type == ToolInvokeMessage.MessageType.TEXT and isinstance(
                    result.message, ToolInvokeMessage.TextMessage
                ):
                    if result.message.text.strip():
                        texts.append(result.message.text)
            if not has_image:
                error_message = "Credential validation image generation failed: no image was returned."
                if texts:
                    error_message += " Tool response: " + "\n".join(texts)
                raise ValueError(error_message)
        except Exception as error:
            raise ToolProviderCredentialValidationError(str(error)) from error
