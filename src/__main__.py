from src.parsing.file_paths import FilePaths
from src.parsing.parse_args import parse_args
from src.parsing.parse_jsons import Jsons, parse_jsons
from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np


def generate_text():
    llm_model = Small_LLM_Model()
    text = "Give me the capital of Franc"
    tokens = llm_model.encode(text)
    tokens_array = [int(token) for token in tokens[0]]
    for _ in range(10):
        logits = llm_model.get_logits_from_input_ids(tokens_array)
        i_max_prob_token = np.argmax(logits)
        new_token_id = int(i_max_prob_token)
        tokens_array += [new_token_id]
        new_token = llm_model.decode([new_token_id])
        text += new_token

    print(text)


def main():
    #generate_text()
    file_paths: FilePaths = parse_args()
    print(file_paths)
    jsons: Jsons = parse_jsons(file_paths)


if __name__ == "__main__":
    main()
