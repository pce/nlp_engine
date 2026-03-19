import argparse
import requests
import os
import sys
import re

# Standard sources for public domain texts
SOURCES = {
    "novel": [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    ],
    "generic_novel": [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    ],
    "business": [
        "https://www.gutenberg.org/cache/epub/34211/pg34211.txt", # The Art of Money Getting
        "https://www.gutenberg.org/cache/epub/4840/pg4840.txt",   # The Theory of the Leisure Class
    ],
    "philosophy": [
        "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",   # The Republic (Plato)
        "https://www.gutenberg.org/cache/epub/730/pg730.txt",     # Beyond Good and Evil
    ]
}

def clean_gutenberg_text(text):
    """Simple cleanup of Project Gutenberg headers and footers."""
    # Find start and end markers
    start_match = re.search(r"\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)
    end_match = re.search(r"\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text)

    if start_match and end_match:
        text = text[start_match.end():end_match.start()]
    elif start_match:
        text = text[start_match.end():]

    # Remove extra whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def fetch_data(category, output_dir):
    if category not in SOURCES:
        print(f"Error: Unknown category '{category}'. Available: {list(SOURCES.keys())}")
        return

    os.makedirs(output_dir, exist_ok=True)
    combined_file = os.path.join(output_dir, f"{category}_source.txt")

    print(f"[*] Fetching training data for category: {category}")

    with open(combined_file, "w", encoding="utf-8") as f_out:
        for url in SOURCES[category]:
            print(f"    - Downloading: {url}")
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                content = clean_gutenberg_text(response.text)
                f_out.write(content)
                f_out.write("\n\n") # Separator
            except Exception as e:
                print(f"    [!] Failed to download {url}: {e}")

    print(f"[+] Success! Training source created: {combined_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch open-source training data from Project Gutenberg.")
    parser.add_argument("--category", type=str, default="novel", help="Category (novel, business, philosophy, generic_novel)")
    parser.add_argument("--out", type=str, default="data/training", help="Output directory")

    args = parser.parse_args()

    # Ensure dependencies are met
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library not found. Install it with: pip install requests")
        sys.exit(1)

    fetch_data(args.category, args.out)
