//! Deterministic, in-process phonemization using Kokoro's trained token alphabet.
//!
//! English uses sentence-aware US/GB dictionaries and a bounded Rust rule
//! fallback. Other languages use the in-tree compatibility frontend. No external
//! program is invoked and no runtime downloads are performed.

use std::collections::HashMap;

use super::espeak_compat::{canonical_language, text_to_phonemes};
use crate::error::{TtsError, TtsResult};

/// Infer the language from a Kokoro voice name's first letter.
pub fn language_from_voice(voice: &str) -> &'static str {
    match voice.chars().next().map(|ch| ch.to_ascii_lowercase()) {
        Some('a') => "en",
        Some('b') => "en-gb",
        Some('e') => "es",
        Some('f') => "fr",
        Some('h') => "hi",
        Some('i') => "it",
        Some('j') => "ja",
        Some('k') => "ko",
        Some('p') => "pt",
        Some('z') => "zh",
        _ => "en",
    }
}

fn language_code(language: &str) -> TtsResult<&'static str> {
    canonical_language(language).ok_or_else(|| {
        TtsError::TokenizerError(format!(
            "Unsupported Kokoro phonemizer language: {language}"
        ))
    })
}

/// Initialize the immutable dictionaries before serving the first request.
///
/// Optional: normal phonemization initializes lazily. US and GB lexicons are
/// cached independently; both share one POS tagger. Calling this concurrently is
/// safe. Rule-only languages have no initialization work.
pub fn prewarm(language: &str) -> TtsResult<()> {
    match language_code(language)? {
        lang @ ("en-us" | "en-gb") => {
            let _ = super::g2p::engine(lang == "en-gb").convert("");
        }
        "ja" => super::espeak_compat::prewarm_japanese()
            .map_err(|err| TtsError::ModelError(err.to_string()))?,
        _ => {}
    }
    Ok(())
}

/// Normalize compatibility glyphs and filter once against the actual model
/// vocabulary. Preserve Misaki tokens A/I/O/Q/W/Y, affricates, stress, and flaps:
/// they name distinct trained embeddings and must not be expanded to IPA.
fn model_phonemes(phonemes: &str, vocab: &HashMap<String, u32>) -> String {
    let mut output = String::with_capacity(phonemes.len());
    for ch in phonemes.chars() {
        let ch = match ch {
            'g' => 'ɡ',
            'ɝ' => 'ɚ',
            // Compatibility fallback for a lateral absent from the v1 alphabet.
            // x and ʲ ARE trained non-English symbols and must remain intact.
            'ɬ' => 'l',
            // Normalize spacing without introducing sentence boundaries.
            ch if ch.is_whitespace() => ' ',
            '\u{0361}' => continue,
            ch => ch,
        };
        let mut buf = [0; 4];
        if vocab.contains_key(ch.encode_utf8(&mut buf)) && (ch != ' ' || !output.ends_with(' ')) {
            output.push(ch);
        }
    }
    output.trim().to_string()
}

/// Convert text into the exact phoneme symbols expected by Kokoro's vocabulary.
/// Unsupported language codes and inputs without any speakable phonemes fail
/// explicitly instead of silently synthesizing English or punctuation-only audio.
pub fn phonemize(text: &str, language: &str, vocab: &HashMap<String, u32>) -> TtsResult<String> {
    let language = language_code(language)?;
    let joined = if matches!(language, "en-us" | "en-gb") {
        let normalized = super::text_normalization::normalize_numbers(text, language);
        super::g2p::engine(language == "en-gb").convert(&normalized)
    } else {
        text_to_phonemes(text, language, None, true, false)
            .map_err(|err| TtsError::TokenizerError(err.to_string()))?
            .join(" ")
    };
    if joined.chars().any(|ch| ch.is_numeric()) {
        return Err(TtsError::TokenizerError(format!(
            "Unnormalized numeric content in phonemizer output (lang: {language})"
        )));
    }
    let output = model_phonemes(&joined, vocab);
    if !output.chars().any(char::is_alphabetic) {
        return Err(TtsError::TokenizerError(format!(
            "Phonemization produced no speakable tokens (lang: {language})"
        )));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Source: hexgrad/Kokoro-82M config.json, vocab section (v1, n_token=178).
    // The fixture uses the actual trained IDs, not a permissive synthetic alphabet.
    fn vocab() -> HashMap<String, u32> {
        serde_json::from_str(include_str!("fixtures/kokoro-v1-vocab.json")).unwrap()
    }
    fn phones(text: &str, lang: &str) -> String {
        phonemize(text, lang, &vocab()).unwrap()
    }

    #[test]
    fn preserves_trained_diphthong_affricate_and_flap_ids() {
        let vocab = vocab();
        assert_eq!(phones("now", "en"), "nˈW");
        assert_eq!(phones("church", "en"), "ʧˈɜɹʧ");
        let ids: Vec<_> = phones("church", "en")
            .chars()
            .map(|ch| vocab[&ch.to_string()])
            .collect();
        assert_eq!(ids, [133, 156, 87, 123, 133]);
        assert!(phones("hitting", "en").contains('T'));
        assert!(phones("go", "en-gb").contains('Q'));
    }

    #[test]
    fn retains_non_english_fricatives_and_palatalization() {
        let vocab = vocab();
        assert_eq!(vocab["x"], 66);
        assert_eq!(vocab["ʲ"], 164);
        assert_eq!(model_phonemes("xʲɬ", &vocab), "xʲl");
        for word in ["gente", "joven"] {
            let output = phones(word, "es");
            assert!(output.contains('χ'), "{word}: {output}");
            assert_eq!(vocab["χ"], 142);
        }
        assert!(phones("ぎゃ", "ja").contains("ɡj"));
    }

    #[test]
    fn keeps_sentence_context_for_heteronyms_and_articles() {
        assert!(phones("I read a book yesterday", "en").contains("ɹˈɛd"));
        assert!(phones("I will read a book", "en").contains("ɹˈid"));
        let record = phones("They record a record", "en");
        assert!(
            record.contains("ɹəkˈɔɹd") && record.contains("ɹˈɛkəɹd"),
            "{record}"
        );
        assert!(phones("the apple", "en").starts_with("ði "));
    }

    #[test]
    fn english_numbers_currency_and_symbols_are_spoken() {
        for (source, expanded) in [
            ("$12.50", "twelve dollars and fifty cents"),
            ("1,000", "one thousand"),
            ("3.14", "three point one four"),
            ("50%", "fifty percent"),
            ("21st", "twenty first"),
            ("10:03", "ten oh three"),
        ] {
            // Compare without stress: words and numbers can legitimately assign
            // different phrase stresses while spelling the same spoken content.
            let strip = |s: String| s.replace(['ˈ', 'ˌ'], "");
            assert_eq!(
                strip(phones(source, "en")),
                strip(phones(expanded, "en")),
                "{source}"
            );
        }
        assert!(phones("A & B", "en").contains("ænd"));
    }

    #[test]
    fn fallback_does_not_invent_sentence_breaks() {
        let text = phones("my phonemizer works", "en");
        assert!(!text.contains('.'), "{text}");
        assert_eq!(phones("phonemizer.", "en").matches('.').count(), 1);
        // Warm cache and cold lookup must preserve both stress and token content.
        assert_eq!(phones("phonemizer", "en"), phones("phonemizer", "en"));
    }

    #[test]
    fn british_lexicon_covers_words_outside_old_overrides() {
        let us = phones("bath car tomato bird", "en");
        let gb = phones("bath car tomato bird", "en-gb");
        assert_ne!(us, gb);
        assert!(
            us.contains("bˈæθ") && gb.contains("bˈɑːθ"),
            "us={us}, gb={gb}"
        );
        assert!(!phones("car", "en-gb").contains('ɹ'));
        assert!(phones("bird", "en-gb").contains("ɜː"));
    }

    #[test]
    fn normalizes_aliases_and_rejects_unknown_languages() {
        for (alias, canonical, text) in [
            ("ES", "es", "Hola"),
            ("pt-BR", "pt", "Olá"),
            ("fr_FR", "fr", "Bonjour"),
            ("en_GB", "en-gb", "bird"),
        ] {
            assert_eq!(phones(text, alias), phones(text, canonical));
        }
        assert!(phonemize("hello", "typo", &vocab()).is_err());
        for text in ["", "   ", "!!!", "🙂"] {
            assert!(phonemize(text, "en", &vocab()).is_err(), "{text}");
        }
    }

    #[test]
    fn multilingual_numbers_survive_real_vocabulary_filtering() {
        for (lang, word) in [
            ("fr", "trois"),
            ("de", "drei"),
            ("es", "tres"),
            ("it", "tre"),
            ("pt", "três"),
            ("ja", "三"),
            ("zh", "三"),
            ("ko", "삼"),
            ("hi", "तीन"),
        ] {
            assert_eq!(phones("3", lang), phones(word, lang), "{lang}");
            let decimal = phones("3.14", lang);
            assert!(
                decimal.chars().filter(|ch| ch.is_alphabetic()).count() > 1,
                "{lang}: {decimal}"
            );
            assert!(phonemize("...", lang, &vocab()).is_err());
        }
        assert!(phones("12,50 €", "fr").chars().any(char::is_alphabetic));
    }

    #[test]
    fn japanese_requests_can_share_the_immutable_dictionary() {
        prewarm("ja").unwrap();
        let expected = phones("こんにちは世界", "ja");
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let expected = &expected;
                scope.spawn(move || assert_eq!(phones("こんにちは世界", "ja"), *expected));
            }
        });
    }
}
