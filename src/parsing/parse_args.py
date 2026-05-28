import sys
from pydantic import BaseModel, model_validator, Field, ValidationError
from typing_extensions import Self
from src.parsing.file_paths import FilePaths


class InputValidator(BaseModel):
    func_def_file: str = Field(default='data/input/functions_definition.json')
    input_file: str = Field(default='data/input/function_calling_tests.json')
    output_file: str = Field(default='data/output/function_calls.json')

    @model_validator(mode='after')
    def check_json_extension(self) -> Self:
        if not self.func_def_file.endswith('.json'):
            raise ValueError("Function definitions file must "
                             + "have the extension .json")
        if not self.input_file.endswith('.json'):
            raise ValueError("Input file must have the extension .json")
        if not self.output_file.endswith('.json'):
            raise ValueError("Output file must have the extension .json")
        return self

    @model_validator(mode='after')
    def check_duplicate(self) -> Self:
        file_paths = {
            self.func_def_file,
            self.input_file,
            self.output_file,
        }
        if len(file_paths) != 3:
            raise ValueError("Cannot have duplicate file paths")
        return self


def parsing_error(desc: str) -> None:
    print(f"Error parsing args: {desc}\n"
          + "Expected format is:\n"
          + "uv run python -m src \n"
          + "[--functions_definition <function_definition_file>]\n"
          + "[--input <input_file>]\n"
          + "[--output <output_file>]\n",
          file=sys.stderr)
    sys.exit(1)


def build_file_paths(input_validator: InputValidator) -> FilePaths:
    file_paths: FilePaths = FilePaths(
        input_validator.func_def_file,
        input_validator.input_file,
        input_validator.output_file,
    )
    
    return file_paths


def parse_args() -> FilePaths:
    argv = sys.argv

    if len(argv) > 7:
        parsing_error("Too many arguments")

    if len(argv) == 2 and len(argv[1]) == 0:
        default_validator = input_validator = InputValidator()
        return build_file_paths(default_validator)

    if (len(argv) - 1) % 2 == 1:
        parsing_error("Number of arguments must be even")

    valid_params: dict[str, None | str] = {
        "--functions_definition": None,
        "--input": None,
        "--output": None,
    }

    for i in range(1, len(argv)):
        if i % 2 == 1:
            if argv[i] not in valid_params:
                parsing_error("Wrong key")
        else:
            if valid_params[argv[i - 1]] is not None:
                parsing_error("Duplicate key")
            valid_params[argv[i - 1]] = argv[i]

    filled_paths: dict[str, str] = {}
    if valid_params["--functions_definition"] is not None:
        filled_paths["func_def_file"] = valid_params["--functions_definition"]
    if valid_params["--input"] is not None:
        filled_paths["input_file"] = valid_params["--input"]
    if valid_params["--output"] is not None:
        filled_paths["output_file"] = valid_params["--output"]

    try:
        input_validator = InputValidator(**filled_paths)
        return build_file_paths(input_validator)
    except ValidationError as e:
        parsing_error(e.errors()[0]["msg"])
        sys.exit(1)
