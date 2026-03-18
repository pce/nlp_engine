/**
 * @file nlp_engine.cpp
 * @brief Implementation of the NLPEngine class for multilingual NLP processing.
 */

#include "nlp_engine.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

namespace pce::nlp {

NLPEngine::NLPEngine() {
  load_stopwords();
  load_dictionary();
  load_sentiment_lexicon();
}

// ===== Tokenization & Basic Processing =====

/**
 * @fn LanguageProfile NLPEngine::detect_language(const std::string& text)
 */
LanguageProfile NLPEngine::detect_language(const std::string& text) {
 LanguageProfile profile;
 profile.language = "en";
 profile.confidence = 0.0f;

 if (text.empty()) return profile;

 std::map<std::string, int> lang_scores;
 lang_scores["en"] = 0;
 lang_scores["de"] = 0;
 lang_scores["fr"] = 0;

 auto tokens = tokenize(to_lower(text));
 for (const auto& token : tokens) {
   if (std::find(english_stopwords_.begin(), english_stopwords_.end(), token) != english_stopwords_.end()) lang_scores["en"]++;
   if (std::find(german_stopwords_.begin(), german_stopwords_.end(), token) != german_stopwords_.end()) lang_scores["de"]++;
   if (std::find(french_stopwords_.begin(), french_stopwords_.end(), token) != french_stopwords_.end()) lang_scores["fr"]++;
 }

 std::string best_lang = "en";
 int max_score = -1;
 int total_hits = 0;

 for (auto const& [lang, score] : lang_scores) {
   total_hits += score;
   if (score > max_score) {
     max_score = score;
     best_lang = lang;
   }
 }

 profile.language = (total_hits > 0) ? best_lang : "en";
 profile.confidence = (total_hits > 0) ? (float)max_score / total_hits : 0.5f;

 return profile;
}

/**
 * @fn SentimentResult NLPEngine::analyze_sentiment(const std::string& text, const std::string& language)
 */
SentimentResult NLPEngine::analyze_sentiment(const std::string& text, const std::string& language) {
  SentimentResult result{.score = 0.0f, .label = "neutral", .confidence = 0.0f};
  if (text.empty()) return result;

  auto tokens = tokenize(to_lower(text));
  int pos_hits = 0;
  int neg_hits = 0;

  for (const auto& token : tokens) {
    if (positive_words_.count(token)) pos_hits++;
    if (negative_words_.count(token)) neg_hits++;
  }

  int total_hits = pos_hits + neg_hits;
  if (total_hits > 0) {
    result.score = (float)(pos_hits - neg_hits) / total_hits;
    if (result.score > 0.1f) result.label = "positive";
    else if (result.score < -0.1f) result.label = "negative";
    result.confidence = std::min(1.0f, (float)total_hits / tokens.size() * 2.0f);
  }

  return result;
}

/**
 * @fn ToxicityResult NLPEngine::detect_toxicity(const std::string& text, const std::string& language)
 */
ToxicityResult NLPEngine::detect_toxicity(const std::string& text, const std::string& language) {
  ToxicityResult result{.is_toxic = false, .score = 0.0f, .category = "none"};
  std::string lower_text = to_lower(text);

  for (const auto& pattern : toxic_patterns_) {
    if (lower_text.find(pattern) != std::string::npos) {
      result.is_toxic = true;
      result.triggers.push_back(pattern);
      result.score = std::min(1.0f, result.score + 0.4f);
      result.category = "offensive";
    }
  }

  return result;
}
// ============ Tokenization & Basic Processing ============

/**
 * @fn std::vector<std::string> NLPEngine::tokenize(const std::string& text)
 */
std::vector<std::string> NLPEngine::tokenize(const std::string& text) {
  std::vector<std::string> tokens;
  std::istringstream iss(text);
  std::string word;

  while (iss >> word) {
    // Remove punctuation
    word.erase(
      std::remove_if(word.begin(), word.end(),
                     [](unsigned char c) { return std::ispunct(c); }),
      word.end()
    );

    if (!word.empty()) {
      tokens.push_back(to_lower(word));
    }
  }

  return tokens;
}

/**
 * @fn std::vector<std::string> NLPEngine::split_sentences(const std::string& text)
 */
std::vector<std::string> NLPEngine::split_sentences(const std::string& text) {
  std::vector<std::string> sentences;
  std::string sentence;
  bool in_quote = false;

  for (size_t i = 0; i < text.length(); ++i) {
    char c = text[i];
    sentence += c;

    if (c == '"') {
      in_quote = !in_quote;
    }

    // End of sentence markers
    if (!in_quote && (c == '.' || c == '!' || c == '?')) {
      // Skip if followed by abbreviation
      if (i + 1 < text.length() && std::isdigit(text[i + 1])) {
        continue;
      }

      // Trim whitespace
      sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
      sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);

      if (!sentence.empty()) {
        sentences.push_back(sentence);
      }
      sentence.clear();
    }
  }

  // Add last sentence if any
  if (!sentence.empty()) {
    sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
    sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);
    if (!sentence.empty()) {
      sentences.push_back(sentence);
    }
  }

  return sentences;
}

/**
 * @fn std::vector<std::string> NLPEngine::remove_stopwords(const std::vector<std::string>& tokens, const std::string& language)
 */
std::vector<std::string> NLPEngine::remove_stopwords(
  const std::vector<std::string>& tokens,
  const std::string& language
) {
  std::vector<std::string> filtered;
  auto stopwords = get_stopwords(language);

  std::unordered_set<std::string> stop_set(stopwords.begin(), stopwords.end());

  for (const auto& token : tokens) {
    if (stop_set.find(token) == stop_set.end()) {
      filtered.push_back(token);
    }
  }

  return filtered;
}

/**
 * @fn std::string NLPEngine::normalize(const std::string& text)
 */
std::string NLPEngine::normalize(const std::string& text) {
  std::string normalized = to_lower(text);
  normalized = remove_punctuation(normalized);
  return normalized;
}

// ============ Spell Checking ============

/**
 * @fn std::vector<Correction> NLPEngine::spell_check(const std::string& text, const std::string& language)
 */
std::vector<Correction> NLPEngine::spell_check(
  const std::string& text,
  const std::string& language
) {
  std::vector<Correction> corrections;
  auto tokens = tokenize(text);

  const std::vector<std::string>* current_dict = &english_dictionary_;
  if (language == "de") current_dict = &german_dictionary_;
  else if (language == "fr") current_dict = &french_dictionary_;

  std::unordered_set<std::string> dict_set(
    current_dict->begin(),
    current_dict->end()
  );

  for (const auto& token : tokens) {
    // Check if word is in dictionary
    if (dict_set.find(token) == dict_set.end() && token.length() > 1) {
      auto suggestions = get_spelling_suggestions(token, 2, language);

      if (!suggestions.empty()) {
        Correction correction{
          .original = token,
          .suggested = suggestions[0],
          .confidence = 1.0f - (levenshtein_distance(token, suggestions[0]) * 0.1f),
          .reason = "Not in dictionary"
        };
        corrections.push_back(correction);
      }
    }
  }

  return corrections;
}

/**
 * @fn std::vector<std::string> NLPEngine::get_spelling_suggestions(const std::string& word, int max_distance, const std::string& language)
 */
std::vector<std::string> NLPEngine::get_spelling_suggestions(
  const std::string& word,
  int max_distance,
  const std::string& language
) {
  std::vector<std::pair<std::string, int>> candidates;

  const std::vector<std::string>* current_dict = &english_dictionary_;
  if (language == "de") current_dict = &german_dictionary_;
  else if (language == "fr") current_dict = &french_dictionary_;

  for (const auto& dict_word : *current_dict) {
    int distance = levenshtein_distance(word, dict_word);
    if (distance <= max_distance && distance > 0) {
      candidates.push_back({dict_word, distance});
    }
  }

  // Sort by distance (closest first)
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  std::vector<std::string> suggestions;
  for (size_t i = 0; i < std::min(size_t(3), candidates.size()); ++i) {
    suggestions.push_back(candidates[i].first);
  }

  return suggestions;
}

/**
 * @fn int NLPEngine::levenshtein_distance(const std::string& s1, const std::string& s2)
 */
int NLPEngine::levenshtein_distance(const std::string& s1, const std::string& s2) {
  size_t len1 = s1.length();
  size_t len2 = s2.length();

  std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));

  for (size_t i = 0; i <= len1; ++i) d[i][0] = i;
  for (size_t j = 0; j <= len2; ++j) d[0][j] = j;

  for (size_t i = 1; i <= len1; ++i) {
    for (size_t j = 1; j <= len2; ++j) {
      int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;

      d[i][j] = std::min({
        d[i - 1][j] + 1,      // deletion
        d[i][j - 1] + 1,      // insertion
        d[i - 1][j - 1] + cost // substitution
      });
    }
  }

  return d[len1][len2];
}

// ============ Summarization ============

SummaryResult NLPEngine::summarize(const std::string& text, float ratio) {
  auto sentences = split_sentences(text);
  auto tfidf = calculate_tfidf(text);

  // Score each sentence
  std::vector<std::pair<size_t, float>> sentence_scores;

  for (size_t i = 0; i < sentences.size(); ++i) {
    float score = calculate_sentence_score(sentences[i], tfidf);
    sentence_scores.push_back({i, score});
  }

  // Select top sentences by ratio
  size_t num_sentences = std::max(size_t(1), static_cast<size_t>(sentences.size() * ratio));
  std::sort(sentence_scores.begin(), sentence_scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::vector<size_t> selected;
  for (size_t i = 0; i < num_sentences && i < sentence_scores.size(); ++i) {
    selected.push_back(sentence_scores[i].first);
  }

  // Sort back to original order
  std::sort(selected.begin(), selected.end());

  // Build summary
  std::string summary;
  for (size_t idx : selected) {
    if (!summary.empty()) summary += " ";
    summary += sentences[idx];
  }

  int original_length = text.length();
  int summary_length = summary.length();

  return SummaryResult{
    .summary = summary,
    .selected_sentences = selected,
    .ratio = ratio,
    .original_length = original_length,
    .summary_length = summary_length
  };
}

/**
 * @fn std::map<std::string, float> NLPEngine::calculate_tfidf(const std::string& text)
 */
std::map<std::string, float> NLPEngine::calculate_tfidf(const std::string& text) {
  auto sentences = split_sentences(text);
  auto tokens = tokenize(text);
  auto filtered = remove_stopwords(tokens);

  // Term frequency
  std::unordered_map<std::string, int> term_count;
  for (const auto& token : filtered) {
    term_count[token]++;
  }

  // Calculate TF-IDF
  std::map<std::string, float> tfidf;
  int total_terms = filtered.size();
  int total_sentences = sentences.size();

  for (const auto& [term, count] : term_count) {
    float tf = static_cast<float>(count) / total_terms;
    float idf = std::log(static_cast<float>(total_sentences) / (count + 1));
    tfidf[term] = tf * idf;
  }

  return tfidf;
}

// ============ Keyword Extraction ============

/**
 * @fn std::vector<Keyword> NLPEngine::extract_keywords(const std::string& text, int max_keywords, const std::string& language)
 */
std::vector<Keyword> NLPEngine::extract_keywords(
  const std::string& text,
  int max_keywords,
  const std::string& language
) {
  auto tokens = tokenize(text);
  auto filtered = remove_stopwords(tokens, language);
  auto tf = calculate_term_frequency(filtered);
  auto tfidf = calculate_tfidf(text);

  std::vector<Keyword> keywords;

  for (const auto& [term, freq] : tf) {
    Keyword kw{
      .term = term,
      .frequency = freq,
      .tfidf_score = tfidf[term],
      .pos = "noun" // Simplified: would need POS tagger
    };
    keywords.push_back(kw);
  }

  // Sort by TF-IDF score
  std::sort(keywords.begin(), keywords.end(),
            [](const auto& a, const auto& b) { return a.tfidf_score > b.tfidf_score; });

  // Return top N
  if (keywords.size() > static_cast<size_t>(max_keywords)) {
    keywords.resize(max_keywords);
  }

  return keywords;
}

/**
 * @fn std::vector<std::string> NLPEngine::extract_terminology(const std::string& text, const std::string& language)
 */
std::vector<std::string> NLPEngine::extract_terminology(
  const std::string& text,
  const std::string& language
) {
  // Simple N-gram terminology extraction (Bigrams/Trigrams)
  std::vector<std::string> terms;
  auto tokens = tokenize(text);
  if (tokens.size() < 2) return terms;

  for (size_t i = 0; i < tokens.size() - 1; ++i) {
    // Basic heuristic: two capitalized words or specific patterns
    if (std::isupper(tokens[i][0]) && std::isupper(tokens[i+1][0])) {
      terms.push_back(tokens[i] + " " + tokens[i+1]);
    }
  }
  return terms;
}

/**
 * @fn std::map<std::string, float> NLPEngine::calculate_term_frequency(const std::vector<std::string>& tokens)
 */
std::map<std::string, float> NLPEngine::calculate_term_frequency(
  const std::vector<std::string>& tokens
) {
  std::map<std::string, float> tf;
  int total = tokens.size();

  std::unordered_map<std::string, int> counts;
  for (const auto& token : tokens) {
    counts[token]++;
  }

  for (const auto& [term, count] : counts) {
    tf[term] = static_cast<float>(count) / total;
  }

  return tf;
}

// ============ Entity Extraction ============

/**
 * @fn std::vector<Entity> NLPEngine::extract_entities(const std::string& text, const std::string& language)
 */
std::vector<Entity> NLPEngine::extract_entities(const std::string& text, const std::string& language) {
  std::vector<Entity> entities;

  // Pattern-based
  auto emails = extract_emails(text);
  auto urls = extract_urls(text);
  auto dates = extract_dates(text);

  entities.insert(entities.end(), emails.begin(), emails.end());
  entities.insert(entities.end(), urls.begin(), urls.end());
  entities.insert(entities.end(), dates.begin(), dates.end());

  // Heuristic for Proper Names (Eigennamenerkennung)
  auto tokens = tokenize(text);
  for (size_t i = 1; i < tokens.size(); ++i) {
    if (tokens[i].length() > 2 && std::isupper(tokens[i][0])) {
      // Basic check: if capitalized but not at start of sentence, likely a Proper Noun
      // This is a simplification for the demo
      Entity name;
      name.text = tokens[i];
      name.type = "proper_noun";
      name.confidence = 0.6f;
      entities.push_back(name);
    }
  }

  return entities;
}

/**
 * @fn std::vector<std::pair<std::string, std::string>> NLPEngine::pos_tag(const std::vector<std::string>& tokens, const std::string& language)
 */
std::vector<std::pair<std::string, std::string>> NLPEngine::pos_tag(
  const std::vector<std::string>& tokens,
  const std::string& language
) {
  std::vector<std::pair<std::string, std::string>> tagged;
  for (const auto& token : tokens) {
    std::string tag = "NN"; // Default Noun

    // Heuristics for ICALL
    if (is_stopword(to_lower(token), language)) tag = "DET";
    else if (token.length() > 3 && (token.substr(token.length()-2) == "ly")) tag = "ADV";
    else if (token.length() > 3 && (token.substr(token.length()-3) == "ing")) tag = "VBG";

    tagged.push_back({token, tag});
  }
  return tagged;
}

/**
 * @fn std::string NLPEngine::stem(const std::string& word, const std::string& language)
 */
std::string NLPEngine::stem(const std::string& word, const std::string& language) {
  std::string s = to_lower(word);
  if (s.length() <= 3) return s;

  // Simple S-Suffix stripping (Grundformreduktion / Stemming)
  if (language == "en") {
    if (s.back() == 's') {
      if (s.size() > 4 && s.substr(s.size()-3) == "ies") return s.substr(0, s.size()-3) + "y";
      return s.substr(0, s.size()-1);
    }
  } else if (language == "de") {
    // Minimal German stemming (removing 'en', 'e', 'er')
    if (s.size() > 5 && s.substr(s.size()-2) == "en") return s.substr(0, s.size()-2);
  }

  return s;
}

std::vector<Entity> NLPEngine::extract_emails(const std::string& text) {
  std::vector<Entity> emails;
  std::regex email_regex(R"(([a-zA-Z0-9._%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}))");
  std::smatch match;

  std::string::const_iterator search_start(text.cbegin());
  while (std::regex_search(search_start, text.cend(), match, email_regex)) {
    Entity email{
      .text = match[0].str(),
      .type = "email",
      .position = static_cast<size_t>(std::distance(text.cbegin(), match[0].first)),
      .confidence = 0.95f
    };
    emails.push_back(email);
    search_start = match[0].second;
  }

  return emails;
}

std::vector<Entity> NLPEngine::extract_urls(const std::string& text) {
  std::vector<Entity> urls;
  std::regex url_regex(R"((https?://[^\s]+))");
  std::smatch match;

  std::string::const_iterator search_start(text.cbegin());
  while (std::regex_search(search_start, text.cend(), match, url_regex)) {
    Entity url{
      .text = match[0].str(),
      .type = "url",
      .position = static_cast<size_t>(std::distance(text.cbegin(), match[0].first)),
      .confidence = 0.95f
    };
    urls.push_back(url);
    search_start = match[0].second;
  }

  return urls;
}

std::vector<Entity> NLPEngine::extract_dates(const std::string& text) {
  std::vector<Entity> dates;
  // ISO format: YYYY-MM-DD
  std::regex iso_regex(R"((\d{4}-\d{2}-\d{2}))");
  std::smatch match;

  std::string::const_iterator search_start(text.cbegin());
  while (std::regex_search(search_start, text.cend(), match, iso_regex)) {
    Entity date{
      .text = match[0].str(),
      .type = "date",
      .position = static_cast<size_t>(std::distance(text.cbegin(), match[0].first)),
      .confidence = 0.95f
    };
    dates.push_back(date);
    search_start = match[0].second;
  }

  return dates;
}

// ============ Readability Analysis ============

/**
 * @fn ReadabilityMetrics NLPEngine::analyze_readability(const std::string& text)
 */
ReadabilityMetrics NLPEngine::analyze_readability(const std::string& text) {
  auto sentences = split_sentences(text);
  auto tokens = tokenize(text);

  int word_count = tokens.size();
  int sentence_count = sentences.size();

  // Count syllables
  int syllable_count = 0;
  for (const auto& token : tokens) {
    syllable_count += count_syllables(token);
  }

  float fre = flesch_reading_ease(word_count, sentence_count, syllable_count);
  float fkg = flesch_kincaid_grade(word_count, sentence_count, syllable_count);

  std::string complexity;
  if (fkg < 6) complexity = "easy";
  else if (fkg < 12) complexity = "medium";
  else complexity = "hard";

  float avg_sentence_length = word_count > 0 ? static_cast<float>(word_count) / sentence_count : 0;

  std::vector<std::string> suggestions;
  if (avg_sentence_length > 20) {
    suggestions.push_back("Consider shortening sentences (avg: " + std::to_string(static_cast<int>(avg_sentence_length)) + " words)");
  }
  if (fkg > 12) {
    suggestions.push_back("Use simpler vocabulary to improve readability");
  }

  return ReadabilityMetrics{
    .flesch_kincaid_grade = fkg,
    .readability_score = fre,
    .complexity = complexity,
    .suggestions = suggestions,
    .word_count = word_count,
    .sentence_count = sentence_count,
    .avg_sentence_length = avg_sentence_length
  };
}

float NLPEngine::flesch_reading_ease(
  int word_count,
  int sentence_count,
  int syllable_count
) {
  if (word_count == 0 || sentence_count == 0) return 0.0f;

  float score = 206.835f
    - 1.015f * (word_count / static_cast<float>(sentence_count))
    - 84.6f * (syllable_count / static_cast<float>(word_count));

  return std::max(0.0f, std::min(100.0f, score));
}

float NLPEngine::flesch_kincaid_grade(
  int word_count,
  int sentence_count,
  int syllable_count
) {
  if (word_count == 0 || sentence_count == 0) return 0.0f;

  float grade = 0.39f * (word_count / static_cast<float>(sentence_count))
    + 11.8f * (syllable_count / static_cast<float>(word_count))
    - 15.59f;

  return std::max(0.0f, grade);
}

/**
 * @fn int NLPEngine::count_syllables(const std::string& word)
 */
int NLPEngine::count_syllables(const std::string& word) {
  std::string w(word);
  std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) { return std::tolower(c); });
  int count = 0;
  bool previous_was_vowel = false;

  for (char c : w) {
    bool is_vowel = (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');

    if (is_vowel && !previous_was_vowel) {
      count++;
    }
    previous_was_vowel = is_vowel;
  }

  // Adjust for silent e at the end (Common in English, but rarely silent in German/French)
  // This heuristic is primarily for English Flesch-Kincaid calculations.
  // In German, a trailing 'e' is almost always a spoken schwa (e.g., 'Hause', 'Liebe').
  if (w.length() > 2 && w.back() == 'e') {
    // Basic heuristic: only subtract if we are not in a language like German
    // or if the word is very short.
    count--;
  }

  // Ensure at least 1 syllable
  return std::max(1, count);
}

// ============ Document Structure Analysis ============

/**
 * @fn DocumentStructure NLPEngine::analyze_structure(const std::string& text)
 */
DocumentStructure NLPEngine::analyze_structure(const std::string& text) {
  auto tokens = tokenize(text);
  int word_count = tokens.size();
  int reading_time = estimate_reading_time(word_count);
  std::string doc_type = detect_document_type(text);

  return DocumentStructure{
    .doc_type = doc_type,
    .sections = {},
    .headings = {},
    .estimated_reading_time = reading_time,
    .estimated_complexity = 0.5f // Placeholder
  };
}

/**
 * @fn int NLPEngine::estimate_reading_time(int word_count)
 */
int NLPEngine::estimate_reading_time(int word_count) {
  const int avg_wpm = 200; // Average words per minute
  return std::max(1, word_count / avg_wpm);
}

/**
 * @fn std::string NLPEngine::detect_document_type(const std::string& text)
 */
std::string NLPEngine::detect_document_type(const std::string& text) {
  std::string lower_text = to_lower(text);

  if (lower_text.find("dear") != std::string::npos && lower_text.find("sincerely") != std::string::npos) {
    return "letter";
  }
  if (lower_text.find("abstract") != std::string::npos && lower_text.find("conclusion") != std::string::npos) {
    return "paper";
  }
  if (lower_text.find("executive summary") != std::string::npos) {
    return "report";
  }
  if (lower_text.find("introduction") != std::string::npos) {
    return "article";
  }

  return "document";
}

// ============ JSON Serialization ============

json NLPEngine::corrections_to_json(const std::vector<Correction>& corrections) {
  json j = json::array();
  for (const auto& c : corrections) {
    j.push_back({
      {"original", c.original},
      {"suggested", c.suggested},
      {"confidence", c.confidence},
      {"reason", c.reason}
    });
  }
  return j;
}

json NLPEngine::keywords_to_json(const std::vector<Keyword>& keywords) {
  json j = json::array();
  for (const auto& k : keywords) {
    j.push_back({
      {"term", k.term},
      {"frequency", k.frequency},
      {"tfidf_score", k.tfidf_score},
      {"pos", k.pos}
    });
  }
  return j;
}

json NLPEngine::entities_to_json(const std::vector<Entity>& entities) {
  json j = json::array();
  for (const auto& e : entities) {
    j.push_back({
      {"text", e.text},
      {"type", e.type},
      {"position", e.position},
      {"confidence", e.confidence}
    });
  }
  return j;
}

json NLPEngine::readability_to_json(const ReadabilityMetrics& metrics) {
  return json{
    {"flesch_kincaid_grade", metrics.flesch_kincaid_grade},
    {"readability_score", metrics.readability_score},
    {"complexity", metrics.complexity},
    {"suggestions", metrics.suggestions},
    {"word_count", metrics.word_count},
    {"sentence_count", metrics.sentence_count},
    {"avg_sentence_length", metrics.avg_sentence_length}
  };
}

json NLPEngine::sentiment_to_json(const SentimentResult& sentiment) {
  return json{
    {"score", sentiment.score},
    {"label", sentiment.label},
    {"confidence", sentiment.confidence}
  };
}

json NLPEngine::toxicity_to_json(const ToxicityResult& toxicity) {
  return json{
    {"is_toxic", toxicity.is_toxic},
    {"score", toxicity.score},
    {"triggers", toxicity.triggers},
    {"category", toxicity.category}
  };
}

json NLPEngine::summary_to_json(const SummaryResult& summary) {
  return json{
    {"summary", summary.summary},
    {"selected_sentences", summary.selected_sentences},
    {"ratio", summary.ratio},
    {"original_length", summary.original_length},
    {"summary_length", summary.summary_length}
  };
}

json NLPEngine::structure_to_json(const DocumentStructure& structure) {
  return json{
    {"doc_type", structure.doc_type},
    {"sections", structure.sections},
    {"headings", structure.headings},
    {"estimated_reading_time", structure.estimated_reading_time},
    {"estimated_complexity", structure.estimated_complexity}
  };
}

// ============ Private Helpers ============

void NLPEngine::load_stopwords() {
  auto load_file = [](const std::string& path, std::vector<std::string>& target) {
    std::ifstream file(path);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        if (!line.empty()) target.push_back(line);
      }
      return true;
    }
    return false;
  };

  // Try loading from data directory first
  if (!load_file("data/stopwords_en.txt", english_stopwords_)) {
    english_stopwords_ = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "is", "was"};
  }

  if (!load_file("data/stopwords_de.txt", german_stopwords_)) {
    german_stopwords_ = {"der", "die", "das", "und", "in", "von", "zu", "mit", "ist", "im"};
  }

  if (!load_file("data/stopwords_fr.txt", french_stopwords_)) {
    french_stopwords_ = {"le", "de", "un", "et", "a", "en", "que", "ne", "pas", "ce"};
  }
}

void NLPEngine::load_sentiment_lexicon() {
  auto load_words = [](const std::string& path, std::vector<std::string>& target) {
    std::ifstream file(path);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') target.push_back(line);
      }
      return true;
    }
    return false;
  };

  std::vector<std::string> pos_list, neg_list;

  // Load Positive Words
  if (load_words("data/sentiment_positive.txt", pos_list)) {
    for (const auto& w : pos_list) positive_words_[w] = 1.0f;
  } else {
    positive_words_ = {{"good", 1.0f}, {"great", 1.0f}, {"gut", 1.0f}, {"bon", 1.0f}};
  }

  // Load Negative Words
  if (load_words("data/sentiment_negative.txt", neg_list)) {
    for (const auto& w : neg_list) negative_words_[w] = 1.0f;
  } else {
    negative_words_ = {{"bad", 1.0f}, {"terrible", 1.0f}, {"schlecht", 1.0f}, {"mauvais", 1.0f}};
  }

  // Load Toxic Patterns
  if (!load_words("data/toxic_words.txt", toxic_patterns_)) {
    toxic_patterns_ = {"stupid", "idiot", "hate", "dumm", "hassen", "débile"};
  }
}

void NLPEngine::load_dictionary() {
  auto load_file = [](const std::string& path, std::vector<std::string>& target) {
    std::ifstream file(path);
    if (file.is_open()) {
      std::string line;
      while (std::getline(file, line)) {
        if (!line.empty()) target.push_back(line);
      }
      return true;
    }
    return false;
  };

  // English Dictionary
  if (!load_file("data/dictionary_en.txt", english_dictionary_)) {
    english_dictionary_ = {
      "about", "after", "all", "also", "another", "any", "as", "ask",
      "because", "been", "before", "being", "below", "between", "both",
      "but", "by", "can", "come", "could", "did", "different", "do",
      "does", "down", "each", "example", "find", "first", "for", "found",
      "from", "gave", "get", "give", "go", "goes", "gone", "got", "great",
      "group", "had", "has", "have", "having", "help", "her", "here", "high",
      "him", "himself", "his", "how", "if", "into", "is", "it", "its", "just",
      "keep", "kind", "know", "large", "last", "left", "less", "let", "like",
      "line", "long", "look", "made", "make", "many", "may", "me", "means",
      "might", "more", "most", "much", "must", "my", "myself", "name", "new",
      "no", "not", "now", "number", "of", "off", "often", "on", "one", "only",
      "or", "other", "our", "ours", "out", "over", "own", "part", "people",
      "perhaps", "place", "play", "said", "same", "say", "school", "second",
      "see", "seem", "several", "she", "should", "show", "since", "so", "some",
      "something", "special", "state", "still", "such", "sure", "take", "tell",
      "than", "thank", "that", "the", "their", "theirs", "them", "then",
      "there", "these", "they", "thing", "this", "those", "through", "time",
      "to", "told", "too", "took", "town", "try", "under", "unit", "until",
      "upon", "use", "used", "very", "want", "was", "watch", "water", "way",
      "we", "well", "went", "were", "what", "when", "where", "which", "while",
      "who", "whole", "why", "will", "with", "word", "work", "world", "would",
      "write", "written", "year", "yes", "you", "your", "yours"
    };
  }

  // German Dictionary
  if (!load_file("data/dictionary_de.txt", german_dictionary_)) {
      german_dictionary_ = {"beispiel", "haus", "welt", "sprache", "lernen"};
  }

  // French Dictionary
  if (!load_file("data/dictionary_fr.txt", french_dictionary_)) {
      french_dictionary_ = {"exemple", "maison", "monde", "langue", "apprendre"};
  }
}

std::string NLPEngine::to_lower(const std::string& str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

std::string NLPEngine::remove_punctuation(const std::string& str) {
  std::string result;
  std::copy_if(str.begin(), str.end(), std::back_inserter(result),
               [](unsigned char c) { return !std::ispunct(c); });
  return result;
}

bool NLPEngine::is_stopword(const std::string& word, const std::string& language) {
  auto stopwords = get_stopwords(language);
  return std::find(stopwords.begin(), stopwords.end(), word) != stopwords.end();
}

std::vector<std::string> NLPEngine::get_stopwords(const std::string& language) {
  if (language == "de") return german_stopwords_;
  if (language == "fr") return french_stopwords_;
  return english_stopwords_;
}

float NLPEngine::calculate_sentence_score(
  const std::string& sentence,
  const std::map<std::string, float>& word_scores
) {
  auto tokens = tokenize(sentence);
  float score = 0.0f;
  int count = 0;

  for (const auto& token : tokens) {
    auto it = word_scores.find(token);
    if (it != word_scores.end()) {
      score += it->second;
      count++;
    }
  }

  return count > 0 ? score / count : 0.0f;
}

} // namespace pce::nlp
