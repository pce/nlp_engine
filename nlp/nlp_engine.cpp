/**
 * @file nlp_engine.cpp
 * @brief Implementation of NLPModel and NLPEngine classes for ICALL.
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

// ============ NLPModel Implementation ============

bool NLPModel::load_from(const std::string& base_path) {
    current_path_ = base_path;
    if (!current_path_.empty() && current_path_.back() != '/' && current_path_.back() != '\\') {
#ifdef _WIN32
        current_path_ += '\\';
#else
        current_path_ += '/';
#endif
    }

    bool success = true;

    // Load Stopwords
    success &= load_file_to_vec(current_path_ + "stopwords_en.txt", data_.stopwords["en"]);
    success &= load_file_to_vec(current_path_ + "stopwords_de.txt", data_.stopwords["de"]);
    success &= load_file_to_vec(current_path_ + "stopwords_fr.txt", data_.stopwords["fr"]);

    // Load Dictionaries
    success &= load_file_to_vec(current_path_ + "dictionary_en.txt", data_.dictionaries["en"]);
    success &= load_file_to_vec(current_path_ + "dictionary_de.txt", data_.dictionaries["de"]);
    success &= load_file_to_vec(current_path_ + "dictionary_fr.txt", data_.dictionaries["fr"]);

    // Load Sentiment Lexicons
    success &= load_lexicon_to_map(current_path_ + "sentiment_positive.txt", data_.positive_lexicon);
    success &= load_lexicon_to_map(current_path_ + "sentiment_negative.txt", data_.negative_lexicon);

    // Load Toxicity Patterns
    success &= load_file_to_vec(current_path_ + "toxic_words.txt", data_.toxic_patterns);

    // Sync legacy views for backward compatibility
    en_stopwords_ = data_.stopwords["en"];
    de_stopwords_ = data_.stopwords["de"];
    fr_stopwords_ = data_.stopwords["fr"];
    en_dict_ = data_.dictionaries["en"];
    de_dict_ = data_.dictionaries["de"];
    fr_dict_ = data_.dictionaries["fr"];
    positive_words_ = data_.positive_lexicon;
    negative_words_ = data_.negative_lexicon;
    toxic_patterns_ = data_.toxic_patterns;

    is_ready_ = success;
    return is_ready_;
}

const std::vector<std::string>& NLPModel::get_stopwords(const std::string& lang) const {
    auto it = data_.stopwords.find(lang);
    if (it != data_.stopwords.end()) {
        return it->second;
    }
    static const std::vector<std::string> empty;
    return empty;
}

const std::vector<std::string>& NLPModel::get_dictionary(const std::string& lang) const {
    auto it = data_.dictionaries.find(lang);
    if (it != data_.dictionaries.end()) {
        return it->second;
    }
    static const std::vector<std::string> empty;
    return empty;
}

bool NLPModel::load_file_to_vec(const std::string& path, std::vector<std::string>& target) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    target.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            target.push_back(line);
        }
    }
    return true;
}

bool NLPModel::load_lexicon_to_map(const std::string& path, std::map<std::string, float>& target) {
    std::ifstream file(path);
    if (!file.is_open()) return false;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line[0] != '#') {
            target[line] = 1.0f; // Default intensity
        }
    }
    return true;
}

// ============ NLPEngine Implementation ============

NLPEngine::NLPEngine(std::shared_ptr<NLPModel> model) : model_(model) {
    if (!model_ || !model_->is_ready()) {
        // In production, consider throwing an exception or logging
    }
}

LanguageProfile NLPEngine::detect_language(const std::string& text) {
    LanguageProfile profile{.language = "en", .confidence = 0.0f};
    if (text.empty() || !model_) return profile;

    std::map<std::string, int> scores = {{"en", 0}, {"de", 0}, {"fr", 0}};
    auto tokens = tokenize(text);

    // Create unordered sets for O(1) stopword lookup
    std::unordered_set<std::string> en_stops(model_->get_stopwords("en").begin(), model_->get_stopwords("en").end());
    std::unordered_set<std::string> de_stops(model_->get_stopwords("de").begin(), model_->get_stopwords("de").end());
    std::unordered_set<std::string> fr_stops(model_->get_stopwords("fr").begin(), model_->get_stopwords("fr").end());

    for (const auto& token : tokens) {
        std::string lower_token = to_lower(token);
        if (en_stops.count(lower_token)) scores["en"]++;
        if (de_stops.count(lower_token)) scores["de"]++;
        if (fr_stops.count(lower_token)) scores["fr"]++;
    }

    std::string best_lang = "en";
    int max_hits = -1;
    int total_hits = 0;

    for (auto const& [lang, count] : scores) {
        total_hits += count;
        if (count > max_hits) {
            max_hits = count;
            best_lang = lang;
        }
    }

    profile.language = (total_hits > 0) ? best_lang : "en";
    profile.confidence = (total_hits > 0) ? (float)max_hits / total_hits : 0.5f;
    return profile;
}

std::vector<std::string> NLPEngine::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        std::string clean = to_lower(word);
        clean.erase(std::remove_if(clean.begin(), clean.end(), [](unsigned char c) { return std::ispunct(c); }), clean.end());
        if (!clean.empty()) tokens.push_back(clean);
    }
    return tokens;
}

std::vector<std::string> NLPEngine::split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::string current;
    bool in_quote = false;
    for (size_t i = 0; i < text.length(); ++i) {
        char c = text[i];
        current += c;
        if (c == '"' || c == '\'') in_quote = !in_quote;
        if (!in_quote && (c == '.' || c == '!' || c == '?')) {
            if (i + 1 == text.length() || std::isspace(text[i + 1])) {
                size_t first = current.find_first_not_of(" \t\n\r");
                if (first != std::string::npos) sentences.push_back(current.substr(first));
                current.clear();
            }
        }
    }
    if (!current.empty()) {
        size_t first = current.find_first_not_of(" \t\n\r");
        if (first != std::string::npos) sentences.push_back(current.substr(first));
    }
    return sentences;
}

std::vector<std::string> NLPEngine::remove_stopwords(const std::vector<std::string>& tokens, const std::string& lang) {
    if (!model_) return tokens;
    const auto& stopwords = model_->get_stopwords(lang);
    std::unordered_set<std::string> stop_set(stopwords.begin(), stopwords.end());
    std::vector<std::string> filtered;
    for (const auto& t : tokens) {
        if (stop_set.find(t) == stop_set.end()) filtered.push_back(t);
    }
    return filtered;
}

std::string NLPEngine::normalize(const std::string& text) {
    return remove_punctuation(to_lower(text));
}

std::vector<Correction> NLPEngine::spell_check(const std::string& text, const std::string& lang) {
    std::vector<Correction> corrections;
    if (!model_) return corrections;
    auto tokens = tokenize(text);
    const auto& dict = model_->get_dictionary(lang);
    std::unordered_set<std::string> dict_set(dict.begin(), dict.end());

    for (const auto& token : tokens) {
        if (token.length() > 1 && dict_set.find(token) == dict_set.end()) {
            auto suggestions = get_spelling_suggestions(token, 2, lang);
            if (!suggestions.empty()) {
                corrections.push_back({token, suggestions[0], 0.8f, "Not in dictionary"});
            }
        }
    }
    return corrections;
}

std::vector<std::string> NLPEngine::get_spelling_suggestions(const std::string& word, int max_dist, const std::string& lang) {
    std::vector<std::string> suggestions;
    if (!model_) return suggestions;
    std::vector<std::pair<std::string, int>> candidates;
    const auto& dict = model_->get_dictionary(lang);

    for (const auto& dw : dict) {
        int d = levenshtein_distance(word, dw);
        if (d <= max_dist) candidates.push_back({dw, d});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    for (size_t i = 0; i < std::min(size_t(3), candidates.size()); ++i) suggestions.push_back(candidates[i].first);
    return suggestions;
}

int NLPEngine::levenshtein_distance(const std::string& s1, const std::string& s2) {
    size_t n = s1.length(), m = s2.length();
    std::vector<std::vector<int>> d(n + 1, std::vector<int>(m + 1));
    for (size_t i = 0; i <= n; ++i) d[i][0] = i;
    for (size_t j = 0; j <= m; ++j) d[0][j] = j;
    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
        }
    }
    return d[n][m];
}

SummaryResult NLPEngine::summarize(const std::string& text, float ratio) {
    auto sentences = split_sentences(text);
    auto tfidf = calculate_tfidf(text);
    std::vector<std::pair<size_t, float>> scores;
    for (size_t i = 0; i < sentences.size(); ++i) scores.push_back({i, calculate_sentence_score(sentences[i], tfidf)});

    std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    size_t count = std::max(size_t(1), size_t(sentences.size() * ratio));
    std::vector<size_t> selected;
    for (size_t i = 0; i < std::min(count, scores.size()); ++i) selected.push_back(scores[i].first);
    std::sort(selected.begin(), selected.end());

    std::string summary;
    for (auto idx : selected) summary += sentences[idx] + " ";
    return {summary, selected, ratio, (int)text.length(), (int)summary.length()};
}

std::map<std::string, float> NLPEngine::calculate_tfidf(const std::string& text) {
    auto sentences = split_sentences(text);
    auto tokens = tokenize(text);
    auto filtered = remove_stopwords(tokens);
    std::unordered_map<std::string, int> term_counts;
    for (const auto& t : filtered) term_counts[t]++;

    std::map<std::string, float> tfidf;
    for (const auto& [term, count] : term_counts) {
        float tf = (float)count / filtered.size();
        int df = 0;
        for (const auto& s : sentences) if (to_lower(s).find(term) != std::string::npos) df++;
        tfidf[term] = tf * std::log((float)sentences.size() / (1 + df));
    }
    return tfidf;
}

std::vector<Keyword> NLPEngine::extract_keywords(const std::string& text, int max_keywords, const std::string& lang) {
    auto tfidf = calculate_tfidf(text);
    std::vector<Keyword> keywords;
    for (const auto& [term, score] : tfidf) keywords.push_back({term, 0.0f, score, ""});
    std::sort(keywords.begin(), keywords.end(), [](const auto& a, const auto& b) { return a.tfidf_score > b.tfidf_score; });
    if (keywords.size() > (size_t)max_keywords) keywords.resize(max_keywords);
    return keywords;
}

std::vector<std::string> NLPEngine::extract_terminology(const std::string& text, const std::string& lang) {
    std::vector<std::string> terms;
    auto tokens = tokenize(text);
    for (size_t i = 0; i < tokens.size() - 1; ++i) {
        if (std::isupper(tokens[i][0]) && std::isupper(tokens[i+1][0])) terms.push_back(tokens[i] + " " + tokens[i+1]);
    }
    return terms;
}

std::vector<std::pair<std::string, std::string>> NLPEngine::pos_tag(const std::vector<std::string>& tokens, const std::string& lang) {
    std::vector<std::pair<std::string, std::string>> tagged;
    if (!model_) return tagged;
    const auto& stops = model_->get_stopwords(lang);
    std::unordered_set<std::string> stop_set(stops.begin(), stops.end());

    for (const auto& t : tokens) {
        std::string tag = "NN";
        if (stop_set.count(to_lower(t))) tag = "DET";
        else if (t.length() > 3 && t.substr(t.length() - 2) == "ly") tag = "ADV";
        tagged.push_back({t, tag});
    }
    return tagged;
}

std::string NLPEngine::stem(const std::string& word, const std::string& lang) {
    std::string s = to_lower(word);
    if (s.length() <= 3) return s;
    if (lang == "en" && s.back() == 's') return s.substr(0, s.length() - 1);
    if (lang == "de" && s.size() > 5 && s.substr(s.size()-2) == "en") return s.substr(0, s.size()-2);
    return s;
}

std::vector<Entity> NLPEngine::extract_entities(const std::string& text, const std::string& lang) {
    std::vector<Entity> entities;
    std::regex email_regex(R"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})");
    std::smatch match;
    std::string::const_iterator search_start(text.cbegin());
    while (std::regex_search(search_start, text.cend(), match, email_regex)) {
        entities.push_back({match[0], "email", (size_t)std::distance(text.cbegin(), match[0].first), 0.95f});
        search_start = match.suffix().first;
    }
    return entities;
}

ReadabilityMetrics NLPEngine::analyze_readability(const std::string& text) {
    auto sentences = split_sentences(text);
    auto tokens = tokenize(text);
    int words = tokens.size(), sents = std::max(1, (int)sentences.size()), syllables = 0;
    for (const auto& t : tokens) syllables += count_syllables(t);

    float score = 206.835f - 1.015f * ((float)words / sents) - 84.6f * ((float)syllables / words);
    float grade = 0.39f * ((float)words / sents) + 11.8f * ((float)syllables / words) - 15.59f;

    return {grade, score, (score > 70 ? "easy" : (score < 40 ? "hard" : "medium")), {}, words, sents, (float)words / sents};
}

SentimentResult NLPEngine::analyze_sentiment(const std::string& text, const std::string& lang) {
    if (!model_) return {0.0f, "neutral", 0.0f};
    auto tokens = tokenize(text);
    int pos = 0, neg = 0;
    const auto& pos_lex = model_->get_positive_lexicon();
    const auto& neg_lex = model_->get_negative_lexicon();
    for (const auto& t : tokens) {
        if (pos_lex.count(t)) pos++;
        if (neg_lex.count(t)) neg++;
    }
    int total = pos + neg;
    if (total == 0) return {0.0f, "neutral", 0.0f};
    float score = (float)(pos - neg) / total;
    return {score, (score > 0.1f ? "positive" : (score < -0.1f ? "negative" : "neutral")), 0.8f};
}

ToxicityResult NLPEngine::detect_toxicity(const std::string& text, const std::string& lang) {
    ToxicityResult res{false, 0.0f, {}, "none"};
    if (!model_) return res;
    std::string lower = to_lower(text);
    for (const auto& p : model_->get_toxic_patterns()) {
        if (lower.find(p) != std::string::npos) {
            res.is_toxic = true;
            res.triggers.push_back(p);
            res.score = std::min(1.0f, res.score + 0.4f);
            res.category = "offensive";
        }
    }
    return res;
}

// --- Internal Helpers ---

std::string NLPEngine::to_lower(const std::string& str) {
    std::string res = str;
    std::transform(res.begin(), res.end(), res.begin(), [](unsigned char c) { return std::tolower(c); });
    return res;
}

std::string NLPEngine::remove_punctuation(const std::string& str) {
    std::string res = str;
    res.erase(std::remove_if(res.begin(), res.end(), [](unsigned char c) { return std::ispunct(c); }), res.end());
    return res;
}

int NLPEngine::count_syllables(const std::string& word) {
    int count = 0;
    bool last_vowel = false;
    std::string w = word;
    std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) { return std::tolower(c); });
    for (char c : w) {
        bool is_vowel = (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y');
        if (is_vowel && !last_vowel) count++;
        last_vowel = is_vowel;
    }
    if (w.length() > 2 && w.back() == 'e') count--;
    return std::max(1, count);
}

float NLPEngine::calculate_sentence_score(const std::string& sentence, const std::map<std::string, float>& scores) {
    auto tokens = tokenize(sentence);
    float sum = 0;
    int count = 0;
    for (const auto& t : tokens) {
        auto it = scores.find(t);
        if (it != scores.end()) { sum += it->second; count++; }
    }
    return count > 0 ? sum / count : 0;
}

// --- Serialization ---

json NLPEngine::corrections_to_json(const std::vector<Correction>& corrections) {
    json j = json::array();
    for (const auto& c : corrections) j.push_back({{"original", c.original}, {"suggested", c.suggested}, {"confidence", c.confidence}, {"reason", c.reason}});
    return j;
}

json NLPEngine::keywords_to_json(const std::vector<Keyword>& keywords) {
    json j = json::array();
    for (const auto& k : keywords) j.push_back({{"term", k.term}, {"tfidf", k.tfidf_score}});
    return j;
}

json NLPEngine::entities_to_json(const std::vector<Entity>& entities) {
    json j = json::array();
    for (const auto& e : entities) j.push_back({{"text", e.text}, {"type", e.type}});
    return j;
}

json NLPEngine::language_to_json(const LanguageProfile& profile) {
    json j;
    j["language"] = profile.language;
    j["confidence"] = profile.confidence;
    return j;
}

json NLPEngine::readability_to_json(const ReadabilityMetrics& metrics) {
    return {{"score", metrics.readability_score}, {"grade", metrics.flesch_kincaid_grade}, {"complexity", metrics.complexity}};
}

json NLPEngine::summary_to_json(const SummaryResult& s) {
    return {{"summary", s.summary}, {"ratio", s.ratio}};
}

json NLPEngine::sentiment_to_json(const SentimentResult& s) {
    return {{"score", s.score}, {"label", s.label}};
}

json NLPEngine::toxicity_to_json(const ToxicityResult& t) {
    return {{"is_toxic", t.is_toxic}, {"score", t.score}, {"category", t.category}};
}

} // namespace pce::nlp
