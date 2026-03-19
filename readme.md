# NLPEngine - Lightweight NLP for ICALL

A lightweight, high-performance C++23 library for multilingual Natural Language Processing, specifically designed for **Intelligent Computer-Assisted Language Learning (ICALL)**.

_Eine leichtgewichtige C++23-Bibliothek für mehrsprachige Computerlinguistik, speziell entwickelt für intelligentes computergestütztes Sprachlernen._

---

## Async Example // Experimental

To do a completely fresh, full build and start the server:

```bash
./dev.sh clean --fastapi
```

To just rebuild and restart without cleaning:

````bash
./dev.sh --fastapi


---

## Key Features / Hauptmerkmale

| Feature               | Description                       | Beschreibung                         |
| :-------------------- | :-------------------------------- | :----------------------------------- |
| **Language ID**       | Automatic detection (EN, DE, FR)  | Automatische Spracherkennung         |
| **Pedagogical Check** | Spell checking & error analysis   | Rechtschreibprüfung & Fehleranalyse  |
| **Readability**       | Flesch-Kincaid complexity scoring | Lesbarkeits- & Komplexitätsanalyse   |
| **Linguistics**       | Rule-based POS tagging & Stemming | Wortartenbestimmung & Stemming       |
| **Ethics & Safety**   | Toxicity & Sentiment analysis     | Toxizitäts- & Stimmungsanalyse       |
| **Terminology**       | Named Entity & Keyword extraction | Eigennamen- & Terminologieextraktion |

---

## Quick Start / Schnelleinstieg (C++23)

The library uses a decoupled architecture where the `NLPModel` manages heavy resources (dictionaries, lexicons) and the `NLPEngine` handles the processing logic.

```cpp
#include "nlp_engine.h"
#include <memory>

using namespace pce::nlp;

int main() {
    // 1. Initialize the Data Model
    auto model = std::make_shared<NLPModel>();
    if (!model->load_from("data")) return 1;

    // 2. Initialize the Engine with the Model
    NLPEngine engine(model);

    std::string text = "Das ist ein Beispiel für Grundformreduktion.";

    // Language Detection / Spracherkennung
    auto profile = engine.detect_language(text); // "de"

    // Stemming / Grundformreduktion
    std::string base = engine.stem("lernen", "de"); // "lern"

    // POS-Tagging / Wortartenbestimmung
    auto tokens = engine.tokenize(text);
    auto tags = engine.pos_tag(tokens, "de");

    // Sentiment & Toxicity / Stimmung & Toxizität
    auto sentiment = engine.analyze_sentiment(text, "de");
    auto toxicity = engine.detect_toxicity("You are stupid!", "en");
}
````

---

## Accuracy & Heuristics Disclaimer

This engine is designed for speed and pedagogical use rather than absolute linguistic perfection. Please note the following heuristics:

### Syllable Counting & 'y' as Vowel

The engine treats **'y'** as a vowel. This is essential for correctly counting syllables in words like _Fly_, _My_, or _System_, where 'y' acts as the nucleus of the syllable.

### Silent 'e' (Language Awareness)

- **English**: Trailing 'e' is often silent (_home_, _fate_) and is subtracted from the syllable count to ensure an accurate Flesch-Kincaid score.
- **German/French**: In these languages, a trailing 'e' often represents a spoken schwa (Schwa-Laut) sound in German (_Liebe_, _Hause_) or influences pronunciation in French (_Route_). Unlike English, it usually constitutes or supports a syllable, so the engine preserves it to maintain accuracy for ICALL applications.

_Note: As a rule-based engine, edge cases in complex morphology may still result in approximations._

---

### Build & Usage

**Prerequisites:** CMake (3.14+), C++17 compiler, Doxygen (optional).

```bash
# 1. Configure and download dependencies
cmake -B build -S .

# 2. Build library, tests, and examples
cmake --build build

# 3. Run the simple example "relative to data dir" by default
cd build && ./nlp_example_simple

# 4. Run the test suite
cd build && ctest --output-on-failure

# 5. Generate Documentation (optional)
cmake --build build --target docs
```
