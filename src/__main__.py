from src.parsing.file_paths import FilePaths
from src.parsing.parse_args import parse_args
from src.parsing.parse_jsons import Jsons, parse_jsons_user, parse_jsons_model
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json


class LLM():
    def __init__(self) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model(
            model_name="Qwen/Qwen3-0.6B"
        )
        self.func_defs: list[dict[str, object]] = []

    def give_context_functions_definitions(
            self,
            func_def: list[dict[str, object]]
        ) -> None:
        self.func_defs = func_def

    def get_chat_template(self, prompt: str) -> str:
        template: str = ("<|im_start|>system\n"
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

    def get_prompt_solution(self, prompt: str) -> str:
        template = self.get_chat_template(prompt)

        tokens = [int(token) for token in self.model.encode(template)[0]]
        
        solution: str = ""
        for _ in range(500):
            logits = self.model.get_logits_from_input_ids(tokens)

            i_max_prob_token = np.argmax(logits)
            new_token_id = int(i_max_prob_token)

            if new_token_id == self.model._tokenizer.eos_token_id:
                break

            tokens += [new_token_id]

            new_token = self.model.decode([new_token_id])
            solution += new_token

        print(solution)
        return solution


def main():
    file_paths: FilePaths = parse_args()
    jsons: Jsons = parse_jsons_user(file_paths)

    llm: LLM = LLM()
    parse_jsons_model(llm.model, jsons)

    llm.give_context_functions_definitions(jsons.func_def)

    print(jsons.vocab)
    _ = llm.get_prompt_solution(jsons.input[0]["prompt"])


if __name__ == "__main__":
    main()
