import sys
from pydantic import BaseModel, ConfigDict, Json, model_validator, Field, ValidationError
from src.parsing.file_paths import FilePaths
from typing import Any, Self
import json


class Jsons():
    def __init__(self) -> None:
        self.func_def: list[dict[str, str]] = []
        self.input: list[dict[str, Any]] = []


def parsing_error(desc: str) -> None:
    print("Error parsing json:\n"
          + f"{desc}\n",
          file=sys.stderr)
    sys.exit(1)


class JsonTypeValidator(BaseModel):
    type: str = Field(min_length=1)


class JsonFuncDefValidator(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, dict[str, JsonTypeValidator]]
    returns: JsonTypeValidator

    @model_validator(mode='after')
    def check_param_name_len(self) -> Self:
        for name in self.parameters:
            if len(name) == 0:
                raise ValueError("Name of parameters cannot be empty")
        return self

    @model_validator(mode='after')
    def check_param_name_dup(self) -> Self:
        seen = set()
        for name in self.parameters:
            if name in seen:
                raise ValueError(f"Duplicate param name detected "
                                 + "in {self.name}")
            seen.add(name)
        return self


class JsonPromptValidator(BaseModel, extra='forbid'):
    prompt: str = Field(min_length=1)


def avoid_dups(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError("Duplicate key detected in json file")
        seen[k] = v
    return seen


def parse_prompts(file_path: str) -> list[dict[str, str]]:
    raw_prompts = []

    try:
        with open(file_path) as f:
            raw_prompts = json.load(f, object_pairs_hook=avoid_dups)
    except Exception as e:
        parsing_error(f"{e}")

    if not isinstance(raw_prompts, list):
        parsing_error("Prompts must be inside an array")

    for prompt in raw_prompts:
        if not isinstance(prompt, dict):
            parsing_error("Every prompt must be a dictionary")

        try:
            _ = JsonPromptValidator(**prompt)
        except Exception as e:
            parsing_error(str(e))

    return raw_prompts


def parse_func_def(file_path: str) -> list[dict[str, Any]]:
    raw_func_defs = []

    try:
        with open(file_path) as f:
            raw_func_defs = json.load(f, object_pairs_hook=avoid_dups)
    except Exception as e:
        parsing_error(f"{e}")

    if not isinstance(raw_func_defs, list):
        parsing_error("Function definitions must be inside an array")

    for func_def in raw_func_defs:
        if not isinstance(func_def, dict):
            parsing_error("Every function definition must be a dictionary")

        try:
            _ = JsonFuncDefValidator(**func_def)
        except Exception as e:
            parsing_error(str(e))

    return raw_func_defs


def parse_jsons(file_paths: FilePaths) -> Jsons:
    jsons = Jsons()
    jsons.func_def = parse_func_def(file_paths.func_def)
    jsons.input = parse_prompts(file_paths.input)
    
    return Jsons()
