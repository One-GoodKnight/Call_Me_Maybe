from src.parsing.file_paths import FilePaths
import json
import os

def write_output(
    file_paths: FilePaths,
    solutions: list[dict[str, object]]
) -> None:
    try:
        os.makedirs(os.path.dirname(file_paths.output), exist_ok=True)
        with open(file_paths.output, 'w') as f:
            json.dump(solutions, f)
    except Exception as e:
        print(f"Error occured writing output: {e}")
