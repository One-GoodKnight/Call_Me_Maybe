from typing import override


class FilePaths():
    def __init__(self, func_def: str, input: str, output: str) -> None:
        self.func_def: str = func_def
        self.input: str = input
        self.output: str = output

    @override
    def __str__(self) -> str:
        return (f"--functions_definition: {self.func_def}\n"
        + f"--input: {self.input}\n"
        + f"--output: {self.output}\n")
