import sys
from pydantic import BaseModel, Field


class InputValidator(BaseModel):
    func_def_file: str = Field(default="data/input/function_definition.json")
    input_file: str = Field(default="data/input/function_calling_tests.json")
    output_file: str = Field(default="data/output/function_calls.json")


class FilePaths():
    def __init__(self, func_def: str, input: str, output: str) -> None:
        self.func_def: str = func_def
        self.input: str = input
        self.output: str = output


def parsing_error() -> None:
    print("Error parsing args, expected format is :\n"
          + "uv run python -m src \n"
          + "[--functions_definition <function_definition_file>]\n"
          + "[--input <input_file>] [–output <output_file>]\n",
          file=sys.stderr)
    sys.exit(1)


def parse_args() -> None:
    argv = sys.argv

    if len(argv) > 3:
        parsing_error()

    for arg in argv:
        parts = arg.split(" ")
        if len(parts) != 2:
            parsing_error()
        param = parts[0]
        value = parts[1]
        if param not in 

    input_validator = InputValidator(first_name="a", last_name="b")
