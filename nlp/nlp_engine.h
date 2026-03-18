#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace pce::nlp {

/**
 * @struct Correction
 * @brief Represents a suggested spelling correction.
 */
struct Correction {
  std::string original;   /**< The original word. */
  std::string suggested;  /**< The suggested replacement. */
  float confidence;       /**< Confidence score (0.0 to 1.0). */
  std::string reason;     /**< Reason for the correction. */
};

/**
 * @struct Keyword
 * @brief Represents an extracted keyword or term.
 */
struct Keyword {
  std::string term;       /**< The keyword string. */
  float frequency;        /**< Raw frequency in text. */
  float tfidf_score;      /**< Calculated TF-IDF score. */
  std::string pos;        /**< Part of speech tag. */
};

/**
 * @struct Entity
 * @brief Represents a named entity (Date, Email, URL, etc.) or proper noun.
 */
struct Entity {
  std::string text;       /**< The extracted text. */
  std::string type;       /**< Entity type (e.g., "email", "url", "proper_noun"). */
  size_t position;        /**< Character position in text. */
  float confidence;       /**< Confidence score. */
};

/**
 * @struct ReadabilityMetrics
 * @brief Detailed readability statistics of a document.
 */
struct ReadabilityMetrics {
  float flesch_kincaid_grade;     /**< US grade level. */
  float readability_score;        /**< Flesch Reading Ease (0-100). */
  std::string complexity;         /**< Label: "easy", "medium", "hard". */
  std::vector<std::string> suggestions; /**< Improvement tips. */
  int word_count;
  int sentence_count;
  float avg_sentence_length;
};

/**
 * @struct SummaryResult
 * @brief Result of text summarization.
 */
struct SummaryResult {
  std::string summary;                /**< The generated summary text. */
  std::vector<size_t> selected_sentences; /**< Indices of original sentences used. */
  float ratio;                        /**< Compression ratio. */
  int original_length;
  int summary_length;
};

/**
 * @struct LanguageProfile
 * @brief Result of language identification.
 */
struct LanguageProfile {
  std::string language;               /**< ISO 639-1 code (en, de, fr). */
  float confidence;                   /**< Identification confidence. */
  std::map<std::string, float> script_distribution; /**< Script analysis. */
};

/**
 * @struct SentimentResult
 * @brief Result of sentiment analysis.
 */
struct SentimentResult {
  float score;                        /**< Sentiment score (-1.0 to 1.0). */
  std::string label;                  /**< "positive", "negative", or "neutral". */
  float confidence;                   /**< Confidence score. */
};

/**
 * @struct ToxicityResult
 * @brief Result of toxicity and offensive language detection.
 */
struct ToxicityResult {
  bool is_toxic;                      /**< Whether the text is considered toxic. */
  float score;                        /**< Toxicity score (0.0 to 1.0). */
  std::vector<std::string> triggers;  /**< Specific words/patterns that triggered detection. */
  std::string category;               /**< "profanity", "hate_speech", "none", etc. */
};

struct DocumentStructure {
 * @brief Analysis of document layout and metadata.
 */
struct DocumentStructure {
  std::string doc_type;               /**< e.g., "report", "article". */
  std::vector<std::string> sections;
  std::vector<std::string> headings;
  int estimated_reading_time;         /**< Minutes. */
  float estimated_complexity;
};

// ============ Main Engine Interface ============

/**
 * @class NLPEngine
 * @brief Core NLP Engine for text processing without heavy ML dependencies.
 *
 * Provides features for tokenization, language detection, spell checking,
 * summarization, keyword extraction, and readability analysis.
 */
class NLPEngine {
public:
  /**
   * @brief Constructs the NLPEngine and loads dictionaries/stopwords.
   */
  explicit NLPEngine();
  ~NLPEngine() = default;

  // ===== Tokenization & Basic Processing =====

  /**
   * Detect the language of the given text.
   * @param text The input text to analyze.
   * @return A LanguageProfile containing the detected ISO code and confidence.
   */
  LanguageProfile detect_language(const std::string& text);

  /**
   * @brief Analyze the sentiment of the text.
   * @param text Input text.
   * @param language ISO language code.
   * @return SentimentResult with score and label.
   */
  SentimentResult analyze_sentiment(const std::string& text, const std::string& language = "en");

  /**
   * @brief Detect toxicity or offensive language.
   * @param text Input text.
   * @param language ISO language code.
   * @return ToxicityResult with detection details.
   */
  ToxicityResult detect_toxicity(const std::string& text, const std::string& language = "en");

  /**
   * @brief Split text into tokens (words).
   * @param text Input string.
   * @return Vector of word tokens.
   */
  std::vector<std::string> tokenize(const std::string& text);

  /**
   * @brief Split text into sentences.
   * @param text Input string.
   * @return Vector of individual sentences.
   */
  std::vector<std::string> split_sentences(const std::string& text);

  /**
   * @brief Remove stop words (common words: the, a, and, etc.) for a specific language.
   * @param tokens Vector of tokens to filter.
   * @param language ISO language code ("en", "de", "fr").
   * @return Filtered vector of tokens.
   */
  std::vector<std::string> remove_stopwords(
    const std::vector<std::string>& tokens,
    const std::string& language = "en"
  );

  /**
   * @brief Convert to lowercase and remove punctuation.
   * @param text Input string.
   * @return Normalized string.
   */
  std::string normalize(const std::string& text);

  // ===== Spell Checking =====

  /**
   * @brief Check spelling and suggest corrections.
   * @param text Input text.
   * @param language ISO language code.
   * @return Vector of Correction objects.
   */
  std::vector<Correction> spell_check(
    const std::string& text,
    const std::string& language = "en"
  );

  /**
   * @brief Get spelling suggestions for a single word.
   * @param word The misspelled word.
   * @param max_distance Maximum Levenshtein distance allowed.
   * @param language ISO language code.
   * @return List of suggested strings.
   */
  std::vector<std::string> get_spelling_suggestions(
    const std::string& word,
    int max_distance = 2,
    const std::string& language = "en"
  );

  /**
   * @brief Calculate Levenshtein distance between two strings.
   */
  static int levenshtein_distance(const std::string& s1, const std::string& s2);

  // ===== Summarization =====

  /**
   * @brief Extract key sentences from text using TF-IDF ranking.
   * @param text Input text.
   * @param ratio Summary length relative to original (0.0 to 1.0).
   * @return SummaryResult containing the summary string.
   */
  SummaryResult summarize(
    const std::string& text,
    float ratio = 0.3
  );

  /**
   * @brief Calculate TF-IDF scores for terms in the document.
   */
  std::map<std::string, float> calculate_tfidf(const std::string& text);

  // ===== Keyword & Terminology Extraction =====

  /**
   * @brief Extract important keywords from text.
   * @param text Input text.
   * @param max_keywords Limit of keywords to return.
   * @param language ISO language code.
   * @return Vector of Keyword structures.
   */
  std::vector<Keyword> extract_keywords(
    const std::string& text,
    int max_keywords = 10,
    const std::string& language = "en"
  );

  /**
   * @brief Extract terminology (multi-word terms).
   * @param text Input text.
   * @param language ISO language code.
   * @return Vector of term strings.
   */
  std::vector<std::string> extract_terminology(
    const std::string& text,
    const std::string& language = "en"
  );

  /**
   * @brief Calculate frequency of each term in tokens.
   */
  std::map<std::string, float> calculate_term_frequency(
    const std::vector<std::string>& tokens
  );

  // ===== Entity Extraction & POS Tagging =====

  /**
   * @brief Perform basic Part-of-Speech tagging based on heuristics.
   * @param tokens List of tokens.
   * @param language ISO language code.
   * @return Vector of pairs {word, tag}.
   */
  std::vector<std::pair<std::string, std::string>> pos_tag(
    const std::vector<std::string>& tokens,
    const std::string& language = "en"
  );

  /**
   * @brief Stemming / Lemmatization (Grundformreduktion).
   * @param word Word to stem.
   * @param language ISO language code.
   * @return Stemmed version of the word.
   */
  std::string stem(const std::string& word, const std::string& language = "en");

  /**
   * @brief Extract named entities from text (Eigennamenerkennung).
   * @param text Input text.
   * @param language ISO language code.
   * @return Vector of Entity structures.
   */
  std::vector<Entity> extract_entities(const std::string& text, const std::string& language = "en");

  /**
   * @brief Extract email addresses using regex.
   */
  std::vector<Entity> extract_emails(const std::string& text);

  /**
   * @brief Extract URLs using regex.
   */
  std::vector<Entity> extract_urls(const std::string& text);

  /**
   * @brief Extract dates using patterns.
   */
  std::vector<Entity> extract_dates(const std::string& text);

  // ===== Readability Analysis =====

  /**
   * @brief Calculate readability metrics including Flesch-Kincaid.
   * @param text Input text.
   * @return ReadabilityMetrics structure.
   */
  ReadabilityMetrics analyze_readability(const std::string& text);

  /**
   * @brief Flesch Reading Ease score calculation.
   */
  float flesch_reading_ease(
    int word_count,
    int sentence_count,
    int syllable_count
  );

  /**
   * @brief Flesch-Kincaid Grade Level calculation.
   */
  float flesch_kincaid_grade(
    int word_count,
    int sentence_count,
    int syllable_count
  );

  /**
   * @brief Count syllables in a word (approximation).
   */
  static int count_syllables(const std::string& word);

  // ===== Document Structure Analysis =====

  /**
   * @brief Analyze document structure and type.
   */
  DocumentStructure analyze_structure(const std::string& text);

  /**
   * @brief Estimate reading time (avg 200 words per minute).
   */
  int estimate_reading_time(int word_count);

  /**
   * @brief Detect document type based on structure.
   */
  std::string detect_document_type(const std::string& text);

  // ===== JSON Serialization =====

  json corrections_to_json(const std::vector<Correction>& corrections);
  json keywords_to_json(const std::vector<Keyword>& keywords);
  json entities_to_json(const std::vector<Entity>& entities);
  json readability_to_json(const ReadabilityMetrics& metrics);
  json summary_to_json(const SummaryResult& summary);
  json structure_to_json(const DocumentStructure& structure);
  json sentiment_to_json(const SentimentResult& sentiment);
  json toxicity_to_json(const ToxicityResult& toxicity);

private:
  // ===== Internal State =====
  std::vector<std::string> english_stopwords_;
  std::vector<std::string> german_stopwords_;
  std::vector<std::string> french_stopwords_;
  std::vector<std::string> english_dictionary_;
  std::vector<std::string> german_dictionary_;
  std::vector<std::string> french_dictionary_;

  // Sentiment Lexicons
  std::map<std::string, float> positive_words_;
  std::map<std::string, float> negative_words_;
  std::vector<std::string> toxic_patterns_;

  // ===== Helpers =====
  void load_stopwords();
  void load_dictionary();
  void load_sentiment_lexicon();

  std::string to_lower(const std::string& str);
  std::string remove_punctuation(const std::string& str);
  bool is_stopword(const std::string& word, const std::string& language);

  std::vector<std::string> get_stopwords(const std::string& language);
  float calculate_sentence_score(
    const std::string& sentence,
    const std::map<std::string, float>& word_scores
  );
};

} // namespace pce::nlp
