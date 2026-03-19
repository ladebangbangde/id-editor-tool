from pathlib import Path

from rembg import remove


class SegmentService:
    def segment_person(self, input_path: str, output_path: str) -> str:
        input_bytes = Path(input_path).read_bytes()
        output_bytes = remove(input_bytes)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_bytes(output_bytes)
        return str(output_file)
