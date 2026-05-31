import sys
from llm_sdk.llm_sdk import Small_LLM_Model
import json
import numpy as np
from typing import cast
from src.parsing.parse_jsons import Jsons, parse_jsons_model
from src.parsing.parse_special_tokens import parse_special_tokens
import math


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

    def __apply_mask(self, logits: list[float], mask: set[int]):
        extended_mask = mask.copy()
        extended_mask.add(self.special_tokens["<|endoftext|>"])

        for i in range(len(logits)):
            if i not in extended_mask:
                logits[i] = -math.inf

    def __get_next_token(
            self,
            tokens: list[int],
            mask: set[int] | None = None) -> int:
        new_token_id = 0

        logits = self.model.get_logits_from_input_ids(tokens)
        if len(logits) == 0:
            self.__no_logits_found()

        if mask:
            self.__apply_mask(logits, mask)

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

    def __get_arguments(self, tokens: list[int], string):
        pass

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

        for _ in range(500):
            new_token_id = self.__get_next_token(tokens)

            if (new_token_id == self.special_tokens["<|endoftext|>"] or
                new_token_id == self.special_tokens["</tool_call>"]):
                break

            tokens += [new_token_id]

            new_token = self.model.decode([new_token_id])
            print(f"{new_token}", end="", flush=True)
            solution += new_token

        print(solution)
        return solution
