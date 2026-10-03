from collections.abc import Mapping


def validate_output_parameters(
    parameters: Mapping[str, object], *, use_v1: bool, is_edit: bool = False
) -> dict[str, object]:
    if not use_v1 and is_edit:
        return {}

    output_format = parameters.get("output_format", "png")
    if not isinstance(output_format, str) or output_format not in {"png", "jpeg"}:
        raise ValueError("Invalid output_format. Choose png or jpeg.")

    if not use_v1:
        return {"output_compression": parameters.get("output_compression", 100)}

    output_parameters: dict[str, str | int] = {"output_format": output_format}
    if output_format == "jpeg":
        compression = parameters.get("output_compression", 100)
        if (
            isinstance(compression, bool)
            or not isinstance(compression, (int, float))
            or not 0 <= compression <= 100
            or int(compression) != compression
        ):
            raise ValueError("Invalid output_compression. Use an integer between 0 and 100.")
        output_parameters["output_compression"] = int(compression)

    return output_parameters


def get_image_mime_type(image_data: bytes, fallback: str) -> str:
    if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return fallback