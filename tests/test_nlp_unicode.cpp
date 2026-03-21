#include <catch2/catch_all.hpp>
#include "nlp/nlp_engine.hh"
#include "nlp/unicode/unicode_utils.hh"
#include <memory>
#include <vector>
#include <string>

using namespace pce::nlp;
using namespace pce::nlp::unicode;

TEST_CASE("Unicode Utilities: Validation and Counting", "[unicode]") {
    SECTION("UTF-8 Validation") {
        std::string valid = "Hello, 世界! 🌍";
        std::string invalid = "Hello \xFF\xFF world"; // Invalid bytes

        CHECK(UnicodeUtils::is_valid_utf8(valid) == true);
        CHECK(UnicodeUtils::is_valid_utf8(invalid) == false);
    }

    SECTION("Code Point Counting") {
        std::string simple = "Hello";
        std::string multi = "Κόσμε"; // Greek 'World' - 5 chars
        std::string emoji = "🚀🚀";   // 2 emojis

        CHECK(UnicodeUtils::count_code_points(simple) == 5);
        CHECK(UnicodeUtils::count_code_points(multi) == 5);
        CHECK(UnicodeUtils::count_code_points(emoji) == 2);
    }
}

TEST_CASE("Unicode Utilities: Script Detection", "[unicode]") {
    CHECK(UnicodeUtils::get_script(U'A') == UnicodeUtils::Script::Latin);
    CHECK(UnicodeUtils::get_script(U'Ω') == UnicodeUtils::Script::Greek);
    CHECK(UnicodeUtils::get_script(U'Я') == UnicodeUtils::Script::Cyrillic);
    CHECK(UnicodeUtils::get_script(U'道') == UnicodeUtils::Script::Han);
    CHECK(UnicodeUtils::get_script(U'あ') == UnicodeUtils::Script::Hiragana);
    CHECK(UnicodeUtils::get_script(U'ア') == UnicodeUtils::Script::Katakana);
}

TEST_CASE("Unicode Utilities: Case Folding", "[unicode]") {
    SECTION("Greek Folding") {
        std::string upper = "ΕΛΛΑΔΑ";
        std::string lower = "ελλαδα";
        CHECK(UnicodeUtils::fold_case(upper) == lower);
    }

    SECTION("Cyrillic Folding") {
        std::string upper = "РОССИЯ";
        std::string lower = "россия";
        CHECK(UnicodeUtils::fold_case(upper) == lower);
    }

    SECTION("Mixed Script Folding") {
        std::string mixed = "Hello Κόσμε Мир";
        std::string expected = "hello κόσμε мир";
        CHECK(UnicodeUtils::fold_case(mixed) == expected);
    }
}

TEST_CASE("NLPEngine: Unicode Tokenization", "[unicode][tokenization]") {
    auto model = std::make_shared<NLPModel>();
    NLPEngine engine(model);

    SECTION("Greek Tokenization") {
        std::string text = "Γεια σου ρε! Τι γίνεται μάγκα;";
        auto tokens = engine.tokenize(text);

        REQUIRE(tokens.size() == 6);
        CHECK(tokens[0] == "γεια");
        CHECK(tokens[1] == "σου");
        CHECK(tokens[2] == "ρε");
        CHECK(tokens[3] == "τι");
        CHECK(tokens[4] == "γίνεται");
        CHECK(tokens[5] == "μάγκα");
    }

    SECTION("Cyrillic Tokenization") {
        std::string text = "Привет, мир. Как дела?";
        auto tokens = engine.tokenize(text);

        REQUIRE(tokens.size() == 4);
        CHECK(tokens[0] == "привет");
        CHECK(tokens[1] == "мир");
        CHECK(tokens[2] == "как");
        CHECK(tokens[3] == "дела");
    }

    SECTION("Mixed Script and Punctuation") {
        std::string text = "Greek: Ω! Cyrillic: Я. Emoji: 🚀.";
        auto tokens = engine.tokenize(text);

        // "greek", "ω", "cyrillic", "я", "emoji", "🚀"
        REQUIRE(tokens.size() == 6);
        CHECK(tokens[0] == "greek");
        CHECK(tokens[1] == "ω");
        CHECK(tokens[5] == "🚀");
    }

    SECTION("Japanese (Basic Boundary Check)") {
        // Japanese text without spaces
        // Since we haven't added the morphological segmenter yet,
        // the generic tokenizer should treat the whole block as one token
        // or split on the ideographic space/punctuation if present.
        std::string text = "こんにちは。世界。"; // Hello. World. (With Japanese full stops)
        auto tokens = engine.tokenize(text);

        // With current implementation, it should split at U+3002 (Japanese Period)
        REQUIRE(tokens.size() == 2);
        CHECK(tokens[0] == "こんにちは");
        CHECK(tokens[1] == "世界");
    }
}

TEST_CASE("NLPEngine: Multilingual Normalization", "[unicode][normalization]") {
    auto model = std::make_shared<NLPModel>();
    NLPEngine engine(model);

    std::string text = "Ω! Привет... Hello.";
    std::string normalized = engine.normalize(text);

    // Normalize currently folds case and replaces punctuation with spaces
    CHECK(normalized.find("ω") != std::string::npos);
    CHECK(normalized.find("привет") != std::string::npos);
    CHECK(normalized.find("hello") != std::string::npos);
}
