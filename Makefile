SRC_DIR		:= src
MYPY_FLAGS	:= 				\
	--warn-return-any		\
	--warn-unused-ignores	\
	--ignore-missing-imports\
	--disallow-untyped-defs	\
	--check-untyped-defs

install:
	uv lock
	uv sync

run:
	uv run python -m $(SRC_DIR)

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .mypy_cache */.mypy_cache */*/.mypy_cache
	rm -rf dist

lint:
	python3 -m flake8 .
	python3 -m mypy . $(MYPY_FLAGS)

lint-strict:
	python3 -m flake8 .
	python3 -m mypy . $(MYPY_FLAGS) --strict
