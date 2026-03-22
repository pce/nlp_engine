/**
 * @file test_nlp_onnx_tokenizer.cpp
 * @brief Unit tests for SimpleTokenizer, ITokenizer, and Encoding.
 *
 * No ONNX Runtime dependency — always compiled regardless of DISABLE_ONNX.
 */

#include <catch2/catch_test_macros.hpp>

#include "addons/onnx/tokenizer.hh"

using namespace pce::nlp::tokenizer;

// ─────────────────────────────────────────────────────────────────────────────
// Encoding — structural invariants
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Encoding — all three vectors have identical length", "[tokenizer]") {
    SimpleTokenizer tok;
    auto enc = tok.encode("the quick brown fox", 32);

    CHECK(enc.input_ids.size() == 32);
    CHECK(enc.attention_mask.size() == 32);
    CHECK(enc.token_type_ids.size() == 32);
}

TEST_CASE("Encoding — CLS and SEP special tokens", "[tokenizer]") {
    SimpleTokenizer tok;
    auto enc = tok.encode("hello world", 16);

    REQUIRE(enc.real_length >= 2);
    CHECK(enc.input_ids.front() == SimpleTokenizer::CLS_ID);
    CHECK(enc.input_ids[enc.real_length - 1] == SimpleTokenizer::SEP_ID);
}

TEST_CASE("Encoding — attention mask marks real vs padding", "[tokenizer]") {
    SimpleTokenizer tok;
    auto enc = tok.encode("hello", 16);

    for (size_t i = 0; i < enc.real_length; ++i)
        CHECK(enc.attention_mask[i] == 1);

    for (size_t i = enc.real_length; i < 16; ++i)
        CHECK(enc.attention_mask[i] == 0);
}

TEST_CASE("Encoding — padding uses PAD_ID", "[tokenizer]") {
    SimpleTokenizer tok;
    auto enc = tok.encode("hi", 16);

    for (size_t i = enc.real_length; i < 16; ++i)
        CHECK(enc.input_ids[i] == SimpleTokenizer::PAD_ID);
}

TEST_CASE("Encoding — empty input produces only CLS and SEP", "[tokenizer]") {
    SimpleTokenizer tok;
    auto enc = tok.encode("", 8);

    CHECK(enc.real_length == 2);
    CHECK(enc.content_length() == 0);
    CHECK_FALSE(enc.empty());
}

TEST_CASE("Encoding — empty() reflects real_length", "[tokenizer]") {
    Encoding enc;
    CHECK(enc.empty());

    enc.real_length = 2;
    CHECK_FALSE(enc.empty());
}

TEST_CASE("Encoding — truncates to max_len", "[tokenizer]") {
    SimpleTokenizer tok;
    // Ten tokens + CLS + SEP would require 12 positions; max_len = 6 forces truncation
    auto enc = tok.encode("one two three four five six seven eight nine ten", 6);

    CHECK(enc.input_ids.size() == 6);
    CHECK(enc.real_length <= 6);
    // First token is always CLS
    CHECK(enc.input_ids.front() == SimpleTokenizer::CLS_ID);
}

// ─────────────────────────────────────────────────────────────────────────────
// SimpleTokenizer — single sentence
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SimpleTokenizer — unknown words map to UNK_ID", "[tokenizer]") {
    SimpleTokenizer tok;  // no vocab loaded
    auto enc = tok.encode("supersecretword", 8);

    // position 0 = CLS, position 1 = content word → UNK
    REQUIRE(enc.real_length >= 2);
    CHECK(enc.input_ids[1] == SimpleTokenizer::UNK_ID);
}

TEST_CASE("SimpleTokenizer — known words map to their vocabulary ID", "[tokenizer]") {
    SimpleTokenizer tok;
    tok.add_token("hello", 10);
    tok.add_token("world", 11);

    auto enc = tok.encode("hello world", 16);
    REQUIRE(enc.real_length == 4);  // CLS hello world SEP
    CHECK(enc.input_ids[1] == 10);
    CHECK(enc.input_ids[2] == 11);
}

TEST_CASE("SimpleTokenizer — case folding before lookup", "[tokenizer]") {
    SimpleTokenizer tok;
    tok.add_token("hello", 10);

    auto enc = tok.encode("HELLO", 8);
    REQUIRE(enc.real_length >= 2);
    CHECK(enc.input_ids[1] == 10);  // uppercased input still finds the lowercase entry
}

TEST_CASE("SimpleTokenizer — vocab_size reflects added tokens", "[tokenizer]") {
    SimpleTokenizer tok;
    CHECK(tok.vocab_size() == 0);
    CHECK_FALSE(tok.is_ready());

    tok.add_token("alpha", 4);
    tok.add_token("beta", 5);

    CHECK(tok.vocab_size() == 2);
    CHECK(tok.is_ready());
}

// ─────────────────────────────────────────────────────────────────────────────
// SimpleTokenizer — sentence pair encoding
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("SimpleTokenizer — encode_pair token_type_ids", "[tokenizer]") {
    SimpleTokenizer tok;
    tok.add_token("hello", 10);
    tok.add_token("world", 11);

    auto enc = tok.encode_pair("hello", "world", 16);

    // sentence A tokens (including leading CLS and its SEP) have type_id 0
    CHECK(enc.token_type_ids[0] == 0);  // CLS

    // sentence B tokens have type_id 1
    bool found_type_1 = false;
    for (size_t i = 0; i < enc.real_length; ++i) {
        if (enc.token_type_ids[i] == 1) {
            found_type_1 = true;
            break;
        }
    }
    CHECK(found_type_1);
}

TEST_CASE("SimpleTokenizer — encode_pair ends with SEP for each segment", "[tokenizer]") {
    SimpleTokenizer tok;
    tok.add_token("a", 10);
    tok.add_token("b", 11);

    // CLS a SEP b SEP  →  real_length = 5
    auto enc = tok.encode_pair("a", "b", 16);
    REQUIRE(enc.real_length == 5);
    CHECK(enc.input_ids[enc.real_length - 1] == SimpleTokenizer::SEP_ID);
}

TEST_CASE("SimpleTokenizer — encode_pair total length respects max_len", "[tokenizer]") {
    SimpleTokenizer tok;
    auto enc = tok.encode_pair(
        "one two three four five",
        "six seven eight nine ten",
        10);

    CHECK(enc.input_ids.size() == 10);
    CHECK(enc.real_length <= 10);
}

// ─────────────────────────────────────────────────────────────────────────────
// ITokenizer — polymorphic usage
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ITokenizer — SimpleTokenizer satisfies interface contract", "[tokenizer]") {
    // Configure vocabulary on the concrete type first, then use through the
    // abstract interface — exactly how ONNXAddon uses it (load vocab, then
    // store as unique_ptr<ITokenizer> for polymorphic encoding calls).
    auto concrete = std::make_unique<SimpleTokenizer>();
    concrete->add_token("test", 4);

    std::unique_ptr<ITokenizer> tok = std::move(concrete);

    auto enc = tok->encode("test", 8);
    CHECK(enc.input_ids.size() == 8);
    CHECK(enc.input_ids[1] == 4);
    CHECK(tok->vocab_size() == 1);
    CHECK(tok->is_ready());
}
