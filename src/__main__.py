from src.parsing.file_paths import FilePaths
from src.parsing.parse_args import parse_args
from src.parsing.parse_jsons import Jsons, parse_jsons
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


class LLM():
    def __init__(self) -> None:
        self.model: Small_LLM_Model = Small_LLM_Model(
            model_name="Qwen/Qwen3-0.6B"
        )
        self.func_defs: list[dict[str, str | dict[str, dict[str, str]]]] = []

    def give_context_functions_definitions(
            self,
            func_def: list[dict[str, str | dict[str, dict[str, str]]]]
        ) -> None:
        self.func_defs = func_def

    def get_prompt_solution(self, prompt: str) -> str:
        conv = [{"role": "user", "content": prompt}]

        text = self.model._tokenizer.apply_chat_template(
            conversation=conv,
            tools=self.func_defs,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        print(text)

        tokens = [int(token) for token in self.model.encode(text)[0]]
        
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
    jsons: Jsons = parse_jsons(file_paths)
    #print(jsons.func_def)
    #print(jsons.input)

    llm: LLM = LLM()
    llm.give_context_functions_definitions(jsons.func_def)
    _ = llm.get_prompt_solution(jsons.input[0]["prompt"])


if __name__ == "__main__":
    main()
