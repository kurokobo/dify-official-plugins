import re
from collections.abc import Mapping


def validate_image_parameters(parameters: Mapping[str, object]) -> dict[str, str]:
    quality = parameters.get("quality", "high")
    if not isinstance(quality, str) or quality not in {"auto", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Invalid quality. Choose auto, low, medium, high, xhigh or max.")

    size = parameters.get("size", "1024x1024")
    if size == "custom":
        size = parameters.get("custom_size")
        if not isinstance(size, str) or not re.fullmatch(r"\d+x\d+", size):
            raise ValueError("Invalid custom_size. Provide a WxH string such as 1024x1024.")
    elif not isinstance(size, str) or size not in {"1024x1024", "1536x1024", "1024x1536", "auto"}:
        raise ValueError("Invalid size. Choose 1024x1024, 1536x1024, 1024x1536, auto or custom.")

    return {"size": size, "quality": quality}
