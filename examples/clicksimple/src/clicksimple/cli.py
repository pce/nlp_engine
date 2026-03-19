import os
import sys
import click
import json
from pathlib import Path
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
# Look for .env in the current directory or parent directories
env_path = Path(".env")
if not env_path.exists():
    # Fallback to look relative to the root if we know where we are
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
    print(f"Error: Could not find the 'nlp_engine' native module. {e}")
    print("Please ensure you built the project with './dev.sh --python'")
    sys.exit(1)

def get_engine():
    """Initialize and return the native engine with default data path"""
    engine = AsyncNLPEngine()
    # Resolve default data path by looking for 'data' directory in the project structure
    # This handles cases where the script might be run from different depths or locations.
    current_path = Path(__file__).resolve()
    data_path = None

    # Traverse up to find the nlp_engine root containing the data folder
    for parent in current_path.parents:
        potential_data = parent / "data"
        if potential_data.is_dir() and (parent / "nlp").exists():
            data_path = str(potential_data)
            break

    if not data_path:
        # Final fallback to expected relative path
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../nlp_engine/data"))

    if not engine.load_model(data_path):
        click.secho(f"Warning: Linguistic data not found at {data_path}", fg="yellow")

    engine.initialize()
    return engine

@click.group()
def main():
    """NLP Studio CLI - Simple Linguistic Interface"""
    pass

@main.command()
@click.argument("text")
def detect(text):
    """Detect the language of the provided text."""
    engine = get_engine()
    try:
        res = engine.process_sync(text, "language")
        data = json.loads(res)

        lang = data.get("language", "Unknown")
        conf = data.get("confidence", 0)

        click.echo(f"Detected: ", nl=False)
        click.secho(f"{lang}", fg="cyan", bold=True, nl=False)
        click.echo(f" (Confidence: {conf}%)")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")

@main.command()
@click.argument("text")
@click.option("--lang", default="en", help="Language for spell checking")
def spell(text, lang):
    """Perform native spell checking on text."""
    engine = get_engine()
    options = {"lang": lang}
    try:
        res = engine.process_sync(text, "spell_check", options)
        data = json.loads(res)

        misspelled = data.get("misspelled", [])
        if not misspelled:
            click.secho("✓ No spelling errors found.", fg="green")
        else:
            click.secho(f"Found {len(misspelled)} errors:", fg="yellow")
            for item in misspelled:
                click.echo(f" - {item}")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")

@main.command()
@click.argument("text")
def sentiment(text):
    """Analyze the sentiment of the provided text."""
    engine = get_engine()
    try:
        res = engine.process_sync(text, "sentiment")
        data = json.loads(res)

        label = data.get("label", "neutral")
        score = data.get("score", 0.0)

        color = "green" if label == "positive" else "red" if label == "negative" else "white"
        click.echo("Sentiment: ", nl=False)
        click.secho(f"{label.upper()}", fg=color, bold=True, nl=False)
        click.echo(f" (Score: {score:.2f})")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")

if __name__ == "__main__":
    main()
