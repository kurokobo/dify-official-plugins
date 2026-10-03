from collections.abc import Mapping
from urllib.parse import urlparse

from openai import AzureOpenAI, OpenAI


def is_v1_api_base(api_base: str) -> bool:
    return urlparse(api_base.strip()).path.rstrip("/").endswith("/openai/v1")


def create_image_client(credentials: Mapping[str, str]) -> AzureOpenAI | OpenAI:
    api_base = credentials["azure_openai_base_url"].strip()
    api_key = credentials["azure_openai_api_key"]
    if is_v1_api_base(api_base):
        return OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/") + "/",
            default_query={"api-version": "preview"},
        )

    api_version = credentials.get("azure_openai_api_version")
    if not api_version or not api_version.strip():
        raise ValueError("API Version is required for dated Azure endpoints. To use v1, set API Base URL to an /openai/v1 endpoint.")
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
        api_version=api_version,
        azure_deployment=credentials["azure_openai_api_model_name"],
    )