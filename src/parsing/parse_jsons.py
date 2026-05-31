import sys
import json
from pydantic import BaseModel, model_validator, Field, ValidationError
from typing import Self
from llm_sdk.llm_sdk import Small_LLM_Model
from src.parsing.file_paths import FilePaths
from typing import cast


class Jsons():
    def __init__(self) -> None:
        self.input: list[dict[str, str]] = []
        self.func_def: list[dict[str, object]] = []
        self.vocab: dict[str, int] = {}
        self.tokenizer: dict[str, object] = {}


def parsing_error(desc: str) -> None:
    print("Error parsing json:\n"
          + f"{desc}\n",
          file=sys.stderr)
    sys.exit(1)


class JsonTypeValidator(BaseModel, extra='forbid'):
    type: str = Field(min_length=1)

    @model_validator(mode='after')
    def check_type(self) -> Self:
        if self.type != "number" and self.type != "string":
            parsing_error(f"'{self.type}' is not a supported argument type")
        return self


class JsonFuncDefValidator(BaseModel, extra='forbid'):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, JsonTypeValidator]
    returns: JsonTypeValidator

    @model_validator(mode='after')
    def check_param_name_len(self) -> Self:
        for name in self.parameters:
            if len(name) == 0:
                raise ValueError("Name of parameters cannot be empty")
        return self

    @model_validator(mode='after')
    def check_param_name_dup(self) -> Self:
        seen: set[str] = set()
        for name in self.parameters:
            if name in seen:
                raise ValueError(f"Duplicate param name detected "
                                 + "in {self.name}")
            seen.add(name)
        return self


class JsonPromptValidator(BaseModel, extra='forbid'):
    prompt: str = Field(min_length=1)


def avoid_dups(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for k, v in pairs:
        if k in seen:
            raise ValueError("Duplicate key detected in json file")
        seen[k] = v
    return seen


def parse_func_def(
        file_path: str
    ) -> list[dict[str, object]]:
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
        except ValidationError as e:
            parsing_error(f"{e.errors()}")

    check_dups: set[str] = set()
    for func in raw_func_defs:
        check_dups.add(cast(str, func["name"]))

    if len(check_dups) != len(raw_func_defs):
        parsing_error("Function overloading is not supported")

    return raw_func_defs


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
        except ValidationError as e:
            parsing_error(f"{e.errors()}")

    return raw_prompts


def parse_jsons_user(file_paths: FilePaths) -> Jsons:
    jsons = Jsons()
    jsons.func_def = parse_func_def(file_paths.func_def)
    jsons.input = parse_prompts(file_paths.input)
    
    return jsons


def parse_jsons_model(model: Small_LLM_Model, jsons: Jsons):
    vocab_path = model.get_path_to_vocab_file()
    tokenizer_path = model.get_path_to_tokenizer_file()
    
    try:
        with open(vocab_path) as f:
            jsons.vocab = json.load(f)
    except Exception as e:
        parsing_error(f"{e}")

    try:
        with open(tokenizer_path) as f:
            jsons.tokenizer = json.load(f)
    except Exception as e:
        parsing_error(f"{e}")
