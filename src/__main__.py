from src.parsing.file_paths import FilePaths
from src.parsing.parse_args import parse_args
from src.parsing.parse_jsons import Jsons, parse_jsons_user, parse_jsons_model
from src.parsing.parse_special_tokens import parse_special_tokens
from src.llm import LLM
import re


def main():
    file_paths: FilePaths = parse_args()
    jsons: Jsons = parse_jsons_user(file_paths)

    llm: LLM = LLM(jsons)

    #llm.give_context_functions_definitions(jsons.func_def)
    
    # print(llm.model.get_path_to_vocab_file())

    _ = llm.get_prompt_solution(jsons.input[0]["prompt"])


if __name__ == "__main__":
    main()
