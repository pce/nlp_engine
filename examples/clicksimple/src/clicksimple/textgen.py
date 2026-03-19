import os
import sys
import click
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# --- Automatic PYTHONPATH discovery ---
def discover_native_module():
    """Attempt to find the nlp_engine binary in common build locations."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        # Check for build folder in project root
        build_dir = parent / "build"
        if build_dir.is_dir():
            # Look for the .so or .pyd file
            matches = list(build_dir.glob("nlp_engine*.so")) + list(build_dir.glob("nlp_engine*.pyd"))
            if matches:
                module_path = str(build_dir)
                if module_path not in sys.path:
                    sys.path.insert(0, module_path)
                return True
    return False

# Try to find the module before importing
discover_native_module()

# --- Native NLP Engine Integration ---
# Load .env to find the native nlp_engine module path (if built via dev.sh)
env_path = Path(".env")
if not env_path.exists():
    # Fallback to look relative to the project structure
    potential_root = Path(__file__).parent.parent.parent.parent.parent
    env_path = potential_root / "build" / ".env"

load_dotenv(dotenv_path=env_path)

if "PYTHONPATH" in os.environ:
    paths = os.environ["PYTHONPATH"].split(os.pathsep)
    for p in reversed(paths):
        if p and p not in sys.path:
            sys.path.insert(0, p)

try:
    from nlp_engine import AsyncNLPEngine
except ImportError as e:
    click.secho(f"Error: Could not find the 'nlp_engine' native module. {e}", fg="red")
    click.echo("Please ensure you built the project with './dev.sh --python'")
    sys.exit(1)

def get_engine_with_model(model_name: str):
    """Initialize and return the native engine with specific Markov model"""
    engine = AsyncNLPEngine()

    # Resolve default data path by looking for 'data' directory in the project structure
    current_path = Path(__file__).resolve()
    data_path = None

    for parent in current_path.parents:
        potential_data = parent / "data"
        if potential_data.is_dir() and (parent / "nlp").exists():
            data_path = str(potential_data)
            break

    if not data_path:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../nlp_engine/data"))

    model_path = os.path.join(data_path, "models", f"{model_name}.json")

    if not engine.load_model(data_path):
        click.secho(f"Warning: Linguistic data not found at {data_path}", fg="yellow")

    if not engine.load_markov_model(model_path, model_name):
        click.secho(f"Error: Could not load Knowledge Pack '{model_name}' from {model_path}", fg="red")
        click.echo("Hint: Run 'nlp_example_train' or use the UI to train the model first.")
        sys.exit(1)

    engine.initialize()
    return engine

@click.group()
def cli():
    """NLP Studio TextGen - Advanced Markov CLI"""
    pass

@cli.command()
@click.argument("seed", default="The")
@click.option("--model", "-m", default="generic_novel", help="Knowledge Pack name (default: generic_novel)")
@click.option("--length", "-l", default=50, type=int, help="Number of words to generate")
@click.option("--temp", "-t", default=1.0, type=float, help="Softmax Temperature (0.1 - 2.0)")
@click.option("--top-p", "-p", default=0.9, type=float, help="Nucleus Sampling threshold (0.0 - 1.0)")
@click.option("--ngram", "-n", default=2, type=int, help="N-Gram context size (2 for Bigram, 3 for Trigram)")
@click.option("--stream/--no-stream", default=True, help="Stream output word-by-word")
@click.option("--benchmark", is_flag=True, help="Print performance statistics")
def generate(seed, model, length, temp, top_p, ngram, stream, benchmark):
    """Generate creative text using the native C++ Markov engine."""

    engine = get_engine_with_model(model)

    options = {
        "length": str(length),
        "temperature": str(temp),
        "top_p": str(top_p),
        "n_gram": str(ngram)
    }

    click.echo(f"[*] Engine: C++ Native | Model: {model} | Seed: '{seed}'")
    click.echo("---")

    start_time = time.perf_counter()
    tokens_received = 0

    try:
        if stream:
            def callback(chunk, is_final):
                nonlocal tokens_received
                if not is_final:
                    click.echo(chunk, nl=False)
                    tokens_received += 1
                else:
                    click.echo("\n")

            # The stream_text call in the native engine is currently blocking for Markov.
            # Wrapping it in a try/except KeyboardInterrupt inside the same process
            # allows the user to break out of the generation loop.
            engine.stream_text(seed, model, callback, options)
        else:
            res = engine.process_sync(seed, model, options)
            click.echo(res)
            tokens_received = length
    except KeyboardInterrupt:
        click.secho("\n\n[!] Generation interrupted. Core bridge closing...", fg="yellow", bold=True)

    end_time = time.perf_counter()
    duration = end_time - start_time

    if benchmark:
        click.echo("---")
        click.secho("Performance Stats:", bold=True, fg="cyan")
        click.echo(f"  Duration:  {duration:.4f} seconds")
        click.echo(f"  Tokens:    {tokens_received}")
        click.echo(f"  Speed:     {tokens_received / duration:.2f} tokens/sec")

@cli.command()
@click.argument("category")
@click.argument("file", type=click.Path(exists=True))
@click.option("--ngram", "-n", default=2, type=int, help="N-Gram size to train")
def train(category, file, ngram):
    """Train a new Knowledge Pack from a local text file."""
    engine = AsyncNLPEngine()

    current_path = Path(__file__).resolve()
    data_path = None
    for parent in current_path.parents:
        potential_data = parent / "data"
        if potential_data.is_dir() and (parent / "nlp").exists():
            data_path = str(potential_data)
            break

    if not data_path:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../nlp_engine/data"))

    output_path = os.path.join(data_path, "models", f"{category}.json")

    click.echo(f"[*] Training model '{category}' from {file}...")

    start_time = time.perf_counter()
    # The native engine wrapper handles the C++ training logic
    success = engine.train_markov_model(str(file), output_path, ngram)
    end_time = time.perf_counter()

    if success:
        click.secho(f"✓ Training complete in {end_time - start_time:.2f}s", fg="green")
        click.echo(f"  Model saved to: {output_path}")
    else:
        click.secho("✗ Training failed in native core.", fg="red")

if __name__ == "__main__":
    cli()
