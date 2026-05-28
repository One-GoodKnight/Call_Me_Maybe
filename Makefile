ARGS = ""

SRC_DIR		:= src
MYPY_FLAGS	:= 					\
	--warn-return-any			\
	--warn-unused-ignores		\
	--ignore-missing-imports	\
	--disallow-untyped-defs		\
	--check-untyped-defs		\

install:
	uv lock
	uv sync

run:
	uv run python -m $(SRC_DIR) $(ARGS)

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name '__pycache__' -exec rm -rv {} +
	find . -type d -name '.mypy_cache' -exec rm -rv {} +

lint:
	python3 -m flake8 .
	python3 -m mypy . $(MYPY_FLAGS)

lint-strict:
	python3 -m flake8 .
	python3 -m mypy . $(MYPY_FLAGS) --strict
