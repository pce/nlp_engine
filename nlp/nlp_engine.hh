/**
 * @file nlp_engine.hh
 * @brief Core NLP processing components and linguistic model management.
 */

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace pce::nlp {

// ============ Data Structures ============

struct Correction {
  std::string original;
  std::string suggested;
  float confidence;
  std::string reason;
};

struct Keyword {
  std::string term;
  float frequency;
  float tfidf_score;
  std::string pos;
};

struct Entity {
  std::string text;
  std::string type;
  size_t position;
  float confidence;
};

struct ReadabilityMetrics {
  float flesch_kincaid_grade;
  float readability_score;
  std::string complexity;
  std::vector<std::string> suggestions;
  int word_count;
  int sentence_count;
  float avg_sentence_length;
};

struct SummaryResult {
  std::string summary;
  std::vector<size_t> selected_sentences;
  float ratio;
  int original_length;
  int summary_length;
};

struct LanguageProfile {
  std::string language;
  float confidence;
  std::map<std::string, float> script_distribution;
};

struct SentimentResult {
  float score;
  std::string label;
  float confidence;
};

struct ToxicityResult {
  bool is_toxic;
  float score;
  std::vector<std::string> triggers;
  std::string category;
};

struct DocumentStructure {
  std::string doc_type;
  std::vector<std::string> sections;
  std::vector<std::string> headings;
  int estimated_reading_time;
  float estimated_complexity;
};

// ============ Data Model ============

/**
 * @class NLPModel
 * @brief Manages the loading and storage of linguistic resources (dictionaries, lexicons).
 *
 * Separating data from logic allows for resource sharing between multiple engines
 * and simplifies cross-platform path management. This class is designed to be
 * read-only after initialization to ensure thread safety across multiple engine instances.
 */
class NLPModel {
public:
  /**
   * @brief Default constructor.
   */
  NLPModel() = default;

  /**
   * @brief Destructor.
   */
  ~NLPModel() = default;

  /**
   * @brief Load all resources from a specific directory.
   *
   * Expects a directory structure containing language-specific files (e.g., 'en_stopwords.txt')
   * and global sentiment lexicons.
   *
   * @param base_path Path to the directory containing .txt resource files.
   * @return true if critical resources were successfully loaded.
   */
  bool load_from(const std::string& base_path);

  /**
   * @brief Retrieves stopword list for a specific language.
   * @param lang Language code (e.g., "en", "de", "fr").
   * @return Vector of stopword strings.
   */
  const std::vector<std::string>& get_stopwords(const std::string& lang) const;

  /**
   * @brief Retrieves dictionary word list for a specific language.
   * @param lang Language code.
   * @return Vector of valid dictionary words.
   */
  const std::vector<std::string>& get_dictionary(const std::string& lang) const;

  /**
   * @brief Gets the positive sentiment lexicon.
   * @return Map of word to positive sentiment score.
   */
  const std::map<std::string, float>& get_positive_lexicon() const { return positive_words_; }

  /**
   * @brief Gets the negative sentiment lexicon.
   * @return Map of word to negative sentiment score.
   */
  const std::map<std::string, float>& get_negative_lexicon() const { return negative_words_; }

  /**
   * @brief Gets patterns used for toxicity detection.
   * @return Vector of regex or keyword patterns.
   */
  const std::vector<std::string>& get_toxic_patterns() const { return toxic_patterns_; }

  /**
   * @brief Checks if the model has been successfully loaded.
   */
  bool is_ready() const { return is_ready_; }

  /**
   * @brief Returns the absolute path from which resources were loaded.
   */
  std::string get_current_path() const { return current_path_; }

  /**
   * @struct DataModel
   * @brief Internal storage for all linguistic resources.
   */
  struct DataModel {
    std::map<std::string, std::vector<std::string>> stopwords;    ///< Map of language codes to lists of stop words.
    std::map<std::string, std::vector<std::string>> dictionaries;  ///< Map of language codes to full dictionary word lists.
    std::map<std::string, float> positive_lexicon;                ///< Map of words to positive sentiment scores.
    std::map<std::string, float> negative_lexicon;                ///< Map of words to negative sentiment scores.
    std::vector<std::string> toxic_patterns;                      ///< List of patterns/words used for toxicity detection.
  };

  const DataModel& get_data() const { return data_; }

private:
  bool is_ready_ = false;
  std::string current_path_;

  DataModel data_;

  // Language Resources (cached view for legacy getters)
  std::vector<std::string> en_stopwords_, de_stopwords_, fr_stopwords_;
  std::vector<std::string> en_dict_, de_dict_, fr_dict_;

  // Sentiment & Toxicity Resources (cached view for legacy getters)
  std::map<std::string, float> positive_words_;
  std::map<std::string, float> negative_words_;
  std::vector<std::string> toxic_patterns_;

  // Internal Loader Helpers
  bool load_file_to_vec(const std::string& path, std::vector<std::string>& target);
  bool load_lexicon_to_map(const std::string& path, std::map<std::string, float>& target);
};

// ============ Processing Engine ============

/**
 * @class NLPEngine
 * @brief Stateless processing logic for NLP tasks.
 *
 * Requires an NLPModel to perform language-aware operations.
 */
/**
 * @class NLPEngine
 * @brief Stateless processing logic for NLP tasks.
 *
 * Requires an NLPModel to perform language-aware operations.
 * Designed for high-performance synchronous execution.
 */
class NLPEngine {
public:
  /**
   * @brief Construct engine with a shared model.
   * @param model Pointer to a loaded NLPModel.
   */
  explicit NLPEngine(std::shared_ptr<NLPModel> model);
  ~NLPEngine() = default;

  // --- Processing Methods ---

  /**
   * @brief Detects the primary language of the input text.
   * @param text Input text.
   * @return LanguageProfile including language code and confidence.
   */
  LanguageProfile detect_language(const std::string& text);

  /**
   * @brief Tokenizes text into individual words or punctuation marks.
   */
  std::vector<std::string> tokenize(const std::string& text);

  /**
   * @brief Tokenizes text while preserving original case and removing punctuation.
   */
  std::vector<std::string> tokenize_with_case(const std::string& text);

  /**
   * @brief Splits a document into individual sentences.
   */
  std::vector<std::string> split_sentences(const std::string& text);

  /**
   * @brief Removes common stop words from a token stream.
   * @param tokens Vector of tokens.
   * @param lang Language code.
   */
  std::vector<std::string> remove_stopwords(const std::vector<std::string>& tokens, const std::string& lang = "en");

  /**
   * @brief Normalizes text (lowercasing, punctuation cleanup).
   */
  std::string normalize(const std::string& text);

  /**
   * @brief Checks spelling and provides corrections for a text.
   */
  std::vector<Correction> spell_check(const std::string& text, const std::string& lang = "en");

  /**
   * @brief Gets similar words for a given misspelled word.
   */
  std::vector<std::string> get_spelling_suggestions(const std::string& word, int max_dist = 2, const std::string& lang = "en");

  /**
   * @brief Calculates edit distance between two strings.
   */
  static int levenshtein_distance(const std::string& s1, const std::string& s2);

  /**
   * @brief Generates an extractive summary of the text.
   * @param ratio Compression ratio (0.1 to 1.0).
   */
  SummaryResult summarize(const std::string& text, float ratio = 0.3);

  /**
   * @brief Calculates term frequency - inverse document frequency scores.
   */
  std::map<std::string, float> calculate_tfidf(const std::string& text);

  /**
   * @brief Extracts the most relevant keywords from a text.
   */
  std::vector<Keyword> extract_keywords(const std::string& text, int max_keywords = 10, const std::string& lang = "en");

  /**
   * @brief Identifies technical or domain-specific terminology using POS and capitalization heuristics.
   */
  std::vector<std::string> extract_terminology(const std::string& text, const std::string& lang = "en");

  /**
   * @brief Performs Part-of-Speech tagging (Noun, Verb, etc.).
   */
  std::vector<std::pair<std::string, std::string>> pos_tag(const std::vector<std::string>& tokens, const std::string& lang = "en");

  /**
   * @brief Reduces a word to its linguistic stem.
   */
  std::string stem(const std::string& word, const std::string& lang = "en");

  /**
   * @brief Extracts named entities (Names, Locations, Dates).
   */
  std::vector<Entity> extract_entities(const std::string& text, const std::string& lang = "en");

  /**
   * @brief Calculates readability scores like Flesch-Kincaid.
   */
  ReadabilityMetrics analyze_readability(const std::string& text);

  /**
   * @brief Analyzes the sentiment of the text (Positive, Negative, Neutral).
   */
  SentimentResult analyze_sentiment(const std::string& text, const std::string& lang = "en");

  /**
   * @brief Detects toxic or offensive content.
   */
  ToxicityResult detect_toxicity(const std::string& text, const std::string& lang = "en");

  // --- Serialization ---

  json corrections_to_json(const std::vector<Correction>& corrections);
  json language_to_json(const LanguageProfile& profile);
  json keywords_to_json(const std::vector<Keyword>& keywords);
  json entities_to_json(const std::vector<Entity>& entities);
  json readability_to_json(const ReadabilityMetrics& metrics);
  json summary_to_json(const SummaryResult& summary);
  json sentiment_to_json(const SentimentResult& sentiment);
  json toxicity_to_json(const ToxicityResult& toxicity);

private:
  std::shared_ptr<NLPModel> model_;

  // Helper Methods
  std::string to_lower(const std::string& str);
  std::string remove_punctuation(const std::string& str);
  static int count_syllables(const std::string& word);
  float calculate_sentence_score(const std::string& sentence, const std::map<std::string, float>& word_scores);
  int estimate_reading_time(int word_count);
  std::string detect_document_type(const std::string& text);
};

} // namespace pce::nlp
