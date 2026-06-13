from src.parsing.file_paths import FilePaths
from src.parsing.parse_args import parse_args
from src.parsing.parse_jsons import Jsons, parse_jsons_user
from src.llm import LLM
from src.output import write_output


def main():
    file_paths: FilePaths = parse_args()
    jsons: Jsons = parse_jsons_user(file_paths)

    llm: LLM = LLM(jsons)

    solutions: list[dict[str, object]] = []

    for prompt in jsons.input:
        solutions.append(llm.get_prompt_solution(prompt["prompt"]))

    write_output(file_paths, solutions)


if __name__ == "__main__":
    main()
