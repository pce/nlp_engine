Ensure you have built the native module first:

```bash
./dev.sh --python
```

## Run the CLI

From the project root set path for the bindings

```bash
export PYTHONPATH=$(pwd)/build
cd examples/clicksimple
uv run python src/clicksimple/cli.py detect "The architect designed a building."

```

Detect language

```bash
uv run python src/clicksimple/cli.py detect "The architect designed a building."
```

# Generate creative text with Trigrams and High Temperature

```bash
uv run python src/clicksimple/textgen.py generate "The system" --ngram 3 --temp 1.5 --benchmark
```

# Trigram generation with temperature

```bash
uv run python src/clicksimple/textgen.py generate "The system" --ngram 3 --temp 1.2
```

## uv

uv init --package
uv add click

// Creating virtual environment at: .venv
