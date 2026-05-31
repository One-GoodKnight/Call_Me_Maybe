import sys
from typing import cast


def parsing_error(desc: str) -> None:
    print("Error parsing tokenizer json:\n"
          + f"{desc}\n",
          file=sys.stderr)
    sys.exit(1)


def find_content(
    added_tokens: list[dict[str, object]],
    token_to_get: str
) -> int:
    id = -1

    for token in added_tokens:
        try:
            if token["content"] == token_to_get:
                id = int(cast(int, token["id"]))
        except Exception as e:
            parsing_error(f"{e}")

    if id == -1:
        parsing_error(f"Could not find {token_to_get} in tokenizer")

    return id


def parse_special_tokens(tokenizer: dict[str, object]) -> dict[str, int]:
    tokens_to_get: list[str] = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<tool_call>", "</tool_call>",
        "<think>", "</think>",
    ]

    added_tokens: list[dict[str, object]] = []
    try:
        added_tokens = cast(list[dict[str, object]], tokenizer["added_tokens"])
    except Exception as e:
        parsing_error(f"{e}")

    special_tokens: dict[str, int] = {}
    for token_to_get in tokens_to_get:
        special_tokens[token_to_get] = find_content(
            added_tokens,
            token_to_get
        )

    return special_tokens
