import sys
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import numpy as np
from typing import cast
from src.parsing.parse_jsons import Jsons, parse_jsons_model
from src.parsing.parse_special_tokens import parse_special_tokens
import math
import re
from enum import Enum


class NumberStage(Enum):
    DOUBLE_QUOTE = 0,
    DOUBLE_POINT = 1,
    SPACE = 2,
    MINUS = 3,
    LEFT_VALUE_DOT = 4,
    DOT = 5,
    RIGHT_VALUE_DOT = 6,
    END = 7,


class LLM():
    def __init__(
            self,
            jsons: Jsons,
        ) -> None:

        self.model: Small_LLM_Model = Small_LLM_Model(
            model_name="Qwen/Qwen3-0.6B"
        )

        parse_jsons_model(self.model, jsons)

        self.func_defs: list[dict[str, object]] = jsons.func_def
        self.vocab: dict[str, int] = jsons.vocab
        self.decoded_vocab: dict[int, str] = {
            v: self.model.decode([v]) for v in self.vocab.values()
        }

        self.special_tokens: dict[str, int] = parse_special_tokens(
            jsons.tokenizer
        )

        self.pre_enc_prompt: dict[str, list[int]] = {}
        self.__calculate_pre_encoded_prompt_tokens()

        self.pre_enc_func_name: set[int] = set()
        self.__calculate_pre_encoded_func_name_tokens()

    def __calculate_pre_encoded_prompt_tokens(self):
        self.pre_enc_prompt |= {'<tool_call>\n': []}
        self.pre_enc_prompt |= {'{"name": "': []}
        self.pre_enc_prompt |= {'", "arguments": {': []}
        self.pre_enc_prompt |= {'}}\n</tool_call>': []}
        self.pre_enc_prompt |= {'"': []}
        self.pre_enc_prompt |= {'</tool_call>\n': []}
        self.pre_enc_prompt |= {',': []}
        self.pre_enc_prompt |= {' ': []}
        self.pre_enc_prompt |= {'}': []}
        for token_text in self.pre_enc_prompt:
            self.pre_enc_prompt[token_text] = (
                [int(token) for token in self.model.encode(token_text)[0]]
            )

    def __calculate_pre_encoded_func_name_tokens(self):
        for func in self.func_defs:
            self.pre_enc_func_name.update(
                [
                    int(token)
                    for token in self.model.encode(
                        cast(str, func["name"])
                    )[0]
                ]
            )

    def __get_chat_template(self, prompt: str) -> str:
        template: str = (f"<|im_start|>system\n"
        + "# Tools\n\n"
        + "You may call one or more functions to assist with the user query.\n"
        + "\nYou are provided with function signatures "
        + "within <tools></tools> XML tags:\n"
        + "<tools>\n")

        for func_dict in self.func_defs:
            template += json.dumps(func_dict) + "\n"

        template += ("</tools>\n\n"
        + "For each function call, return a json object with function "
        + "name and arguments within <tool_call></tool_call> XML tags:\n"
        + "<tool_call>\n"
        + '{"name": <function-name>, "arguments": <args-json-object>}\n'
        + "</tool_call><|im_end|>\n"
        + "<|im_start|>user\n"
        + f"{prompt}<|im_end|>\n"
        + "<|im_start|>assistant\n"
        + "<think>\n\n</think>\n\n"
        )
        return template

    def __no_logits_found(self):
        print("No logit found\n", file=sys.stderr)
        sys.exit(1)

    def __incomplete_prompt_solution(self):
        print("Incomplete solution to prompt\n", file=sys.stderr)
        sys.exit(1)

    def __apply_mask_token_ids(self, logits: list[float], mask: set[int]):
        for i in range(len(logits)):
            if i == self.special_tokens["<|endoftext|>"]:
                continue
            if i not in mask:
                logits[i] = -math.inf

    def __apply_mask_regex(self, logits: list[float], mask: str):
        for i in range(len(logits)):
            if i == self.special_tokens["<|endoftext|>"]:
                continue
            if i not in self.decoded_vocab:
                logits[i] = -math.inf
                continue
            try:
                if not re.match(mask, self.decoded_vocab[i]):
                    logits[i] = -math.inf
            except Exception as e:
                print(f"Error trying to access decoded_vocab {e}")
            # else:
                # print("logit not -inf")
        print(mask)
        # for i in range(len(logits)):
        #     if logits[i] > -100000:
        #         if i in self.decoded_vocab:
        #             print(self.decoded_vocab[i])

    def __get_next_token(
            self,
            tokens: list[int],
            mask: set[int] | str | None = None) -> int:
        # if isinstance(mask, str):
        #     text = self.model.decode(tokens)
        #     print(text)
        new_token_id = 0

        logits = self.model.get_logits_from_input_ids(tokens)
        if len(logits) == 0:
            self.__no_logits_found()

        if isinstance(mask, set):
            self.__apply_mask_token_ids(logits, mask)
        if isinstance(mask, str):
            self.__apply_mask_regex(logits, mask)

        i_max_prob_token = np.argmax(logits)
        new_token_id = int(i_max_prob_token)

        return new_token_id
    
    def __get_function_name(self, tokens: list[int]) -> str:
        max_tokens = 30
        func_name = ""

        mask = self.pre_enc_func_name.copy()
        mask.update(self.pre_enc_prompt['"'])

        for _ in range(max_tokens):
            new_token_id = self.__get_next_token(tokens, mask)

            if new_token_id == self.special_tokens['<|endoftext|>']:
                self.__incomplete_prompt_solution()

            if new_token_id == self.pre_enc_prompt['"'][0]:
                return func_name

            tokens += [new_token_id]

            new_token = self.model.decode([new_token_id])
            func_name += new_token

        self.__incomplete_prompt_solution()

    def __get_argument_template(
            self,
            name: str,
            type: str,
            index: int) -> str:
        template = ""
        if index != 0:
            template += ","
        template += f' "{name}'
        if type == "string":
            template += '": "'
        return template

    def __get_arguments_name_type(self, func_name: str) -> list[dict[str, str]]:
        arg_names: list[dict[str, str]] = []
        for func in self.func_defs:
            if func["name"] == func_name:
                for name, type_dict in cast(dict[str, object], func["parameters"]).items():
                    arg_names.append({name: cast(dict[str, str], type_dict)["type"]})
        return arg_names
    
    def __get_regex_mask(
            self,
            type: str,
            stage: NumberStage = NumberStage.DOUBLE_QUOTE) -> str:
        mask = r""
        if type == "string":
            mask = r'^(?=.)[^"]*"?$'
        elif type == "number":
            if stage == NumberStage.DOUBLE_QUOTE:
                mask = r'^(?=.)"?:? ?-?[0-9]*$'
            if stage == NumberStage.DOUBLE_POINT:
                mask = r'^(?=.):? ?-?[0-9]*$'
            if stage == NumberStage.SPACE:
                mask = r'^(?=.) ?-?[0-9]*$'
            if stage == NumberStage.MINUS:
                mask = r'^(?=.)-?[0-9]*$'
            if stage == NumberStage.LEFT_VALUE_DOT:
                mask = r"^(?=.)[0-9]+$"
            if stage == NumberStage.DOT:
                mask = r"^(?=.)[0-9]*(\.?[0-9]+)?(,| |})?$"
            if stage == NumberStage.RIGHT_VALUE_DOT:
                mask = r"^(?=.)[0-9]+(,| |})?$"
            if stage == NumberStage.END:
                mask = r"^(?=.)[0-9]*(,| |})?$"

        return mask

    def __get_number_value_stage(self, value: str) -> NumberStage:
        if "." in value and value.index(".") + 1 != len(value):
            return NumberStage.END
        if "." in value and value.index(".") + 1 == len(value):
            return NumberStage.RIGHT_VALUE_DOT
        if (len(value) > 2 or (len(value) == 2 and value[0] == "-")
            or (len(value) == 1 and value[0] != "-")):
            return NumberStage.DOT
        if len(value) > 1 or (len(value) == 1 and value[0] == "-"):
            return NumberStage.LEFT_VALUE_DOT

        return NumberStage.DOUBLE_QUOTE

    def __get_arg_value(self, tokens: list[int], type: str) -> str:
        max_tokens = 100
        value = ""
        stage: NumberStage = NumberStage.DOUBLE_QUOTE
        mask: str | None = None
        if type == "string":
            mask = self.__get_regex_mask(type)
        generated_space = False
        
        for _ in range(max_tokens):
            if type == "number":
                if generated_space:
                    stage = self.__get_number_value_stage(value)
                mask = self.__get_regex_mask(type, stage)

            new_token_id = self.__get_next_token(tokens, mask)

            if new_token_id == self.special_tokens['<|endoftext|>']:
                print("End of text generated")
                self.__incomplete_prompt_solution()

            if (new_token_id == self.pre_enc_prompt[','][0]
                or new_token_id == self.pre_enc_prompt[' '][0]
                or new_token_id == self.pre_enc_prompt['}'][0]):
                return value

            new_token = self.model.decode([new_token_id])

            end = False
            if any(char.isdigit() for char in value) and "," in new_token:
                end = True
                new_token = new_token[:new_token.index(",")]
            if any(char.isdigit() for char in value) and " " in new_token:
                end = True
                new_token = new_token[:new_token.index(" ")]
            if any(char.isdigit() for char in value) and "}" in new_token:
                end = True
                new_token = new_token[:new_token.index("}")]
            
            if end:
                new_token_truncated_id = self.model.encode(new_token)[0]
                tokens += [int(token) for token in new_token_truncated_id]
            else:
                tokens += [new_token_id]
            
            if new_token.startswith('"'):
                new_token = new_token[1:]
                stage = NumberStage.DOUBLE_POINT
            if new_token.startswith(':'):
                new_token = new_token[1:]
                stage = NumberStage.SPACE
            if new_token.startswith(' '):
                new_token = new_token[1:]
                stage = NumberStage.MINUS
                generated_space = True
            value += new_token

            print(value)

            if end:
                return value

        self.__incomplete_prompt_solution()

    def __get_arguments(
            self,
            tokens: list[int],
            func_name: str) -> list[dict[str, str]]:
        arg_values: list[dict[str, str]] = []
        args = self.__get_arguments_name_type(func_name)

        for i in range(len(args)):
            name: str = list(args[i].keys())[0]
            type: str = list(args[i].values())[0]
            template = self.__get_argument_template(name, type, i)
            tokens += [int(token) for token in self.model.encode(template)[0]]
            value = self.__get_arg_value(tokens, type)
            arg_values.append({name: value})

        return arg_values

    def __assert_function_name(self, name: str):
        found = False
        for func in self.func_defs:
            if func["name"] == name:
                found = True
        if not found:
            print("Incorrect function name generated", file=sys.stderr)
            sys.exit(1)

    def get_prompt_solution(self, prompt: str) -> str:
        template = self.__get_chat_template(prompt)

        tokens = [int(token) for token in self.model.encode(template)[0]]

        solution: str = ""

        tokens += self.pre_enc_prompt['<tool_call>\n']
        tokens += self.pre_enc_prompt['{"name": "']
        solution += '{"name": "'

        name = self.__get_function_name(tokens)
        self.__assert_function_name(name)
        solution += name

        tokens += self.pre_enc_prompt['", "arguments": {']
        solution += '", "arguments": {'
        
        args = self.__get_arguments(tokens, name)
        print(args)

        print(solution)
        return solution
