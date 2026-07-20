//! Text-to-phoneme conversion for Kokoro-82M.
//!
//! Converts plain text to Kokoro-compatible IPA phoneme strings using an
//! in-tree pure-Rust `espeak-rs` compatibility layer.

use std::collections::HashMap;

use crate::error::{TtsError, TtsResult};

use super::espeak_compat::text_to_phonemes;

fn lang_to_espeak(lang: &str) -> &'static str {
    match lang {
        "en" | "en-us" | "a" => "en-us",
        "en-gb" | "b" => "en-gb",
        "ja" | "j" => "ja",
        "zh" | "z" => "cmn",
        "ko" | "k" => "ko",
        "fr" | "f" => "fr",
        "de" | "d" => "de",
        "it" | "i" => "it",
        "pt" | "p" => "pt",
        "es" | "e" => "es",
        "hi" | "h" => "hi",
        _ => "en-us",
    }
}

/// Infer the language from a Kokoro voice name's first letter.
///
/// Kokoro voice naming convention:
///   first letter = language (a=American, b=British, e=Spanish, f=French,
///   h=Hindi, i=Italian, j=Japanese, p=Portuguese, z=Chinese)
pub fn language_from_voice(voice: &str) -> &'static str {
    match voice.chars().next() {
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

fn apply_kokoro_replacements(phonemes: &str) -> String {
    let mut s = phonemes
        .replace('ʲ', "j")
        .replace('ɝ', "ɚ")
        .replace('g', "ɡ")
        .replace('x', "k")
        .replace("ɬ", "l");

    // Remove tie bars (espeak uses combining tie U+0361 for affricates)
    s = s.replace('\u{0361}', "");

    s
}

/// Filter a phoneme string to only characters present in the Kokoro vocab.
fn filter_to_vocab(phonemes: &str, vocab: &HashMap<String, u32>) -> String {
    phonemes
        .chars()
        .filter(|c| {
            let mut buf = [0; 4];
            vocab.contains_key(c.encode_utf8(&mut buf))
        })
        .collect()
}

fn is_kokoro_vowel(c: char) -> bool {
    matches!(
        c,
        'a' | 'e' | 'i' | 'o' | 'u'
            | 'ɑ' | 'ɐ' | 'ɒ' | 'æ' | 'ɔ' | 'ə' | 'ɚ' | 'ɛ' | 'ɜ' | 'ɨ' | 'ɪ' | 'ɯ' | 'ʌ' | 'ɣ' | 'ɤ' | 'ʊ' | 'ʏ'
            | 'A' | 'I' | 'O' | 'Q' | 'S' | 'T' | 'W' | 'Y' | 'ᵻ'
    )
}

fn map_g2p_to_kokoro_ipa(ph: &str) -> String {
    // voice_g2p maps several standard Arpabet sounds to single-character shorthand tokens.
    // While present in the tokenizer vocabulary JSON, the model weights expect standard
    // multi-character espeak-ng IPA sequences. Translating them here prevents dropouts.
    ph.replace('A', "eɪ")  // EY -> eɪ (e.g. "bait" -> bˈeɪt, "became" -> bəkˈeɪm)
      .replace('I', "aɪ")  // AY -> aɪ (e.g. "bite" -> bˈaɪt, "I" -> ˌaɪ)
      .replace('O', "oʊ")  // OW -> oʊ (e.g. "boat" -> bˈoʊt)
      .replace('W', "aʊ")  // AW -> aʊ (e.g. "now" -> nˈaʊ, "without" -> wɪðˈaʊt)
      .replace('Y', "ɔɪ")  // OY -> ɔɪ (e.g. "boy" -> bˈɔɪ)
      .replace('T', "t")   // DX/T flap-t -> t (e.g. "hitting" -> hˈɪtɪŋ)
      .replace('Q', "juː") // UW/YUW -> juː
      .replace('S', "ʃ")   // SH -> ʃ (e.g. "she" -> ʃi)
      .replace('D', "ð")   // DH -> ð (e.g. "this" -> ðˈɪs)
      .replace('ʧ', "tʃ")  // voiceless postalveolar affricate -> tʃ (e.g. "virtually" -> vˈɜɹtʃəwəli)
      .replace('ʤ', "dʒ")  // voiced postalveolar affricate -> dʒ (e.g. "gin" -> dʒˈɪn)
}

fn phonemize_token_hybrid(word: &str, espeak_lang: &str, vocab: &HashMap<String, u32>) -> String {
    let normalized = word.to_lowercase();
    if normalized == "walkthrough" {
        return "wˈɔkθɹuː".to_string();
    }
    if normalized == "walkthroughs" {
        return "wˈɔkθɹuːz".to_string();
    }

    let mut ph = voice_g2p::english_to_phonemes(word)
        .map(|p| map_g2p_to_kokoro_ipa(&p))
        .ok()
        .filter(|p| !p.trim().is_empty() && p.chars().any(|c| c.is_alphabetic() || vocab.contains_key(&c.to_string())))
        .unwrap_or_else(|| {
            text_to_phonemes(word, espeak_lang, None, true, false)
                .map(|p| p.join(""))
                .unwrap_or_default()
        });

    // Strip leading stress markers from the start of the token ONLY if followed by a consonant.
    // Putting a stress marker before a consonant (e.g. "ˈf") causes Kokoro to mispronounce it as "ah".
    // But stress markers before vowels (e.g. "ˌaɪ", "ˈæ") are required and must be preserved.
    while ph.starts_with('ˈ') || ph.starts_with('ˌ') {
        if let Some(next_char) = ph.chars().nth(1) {
            if !is_kokoro_vowel(next_char) {
                ph.remove(0);
                continue;
            }
        }
        break;
    }
    ph
}

/// Convert plain text to Kokoro-compatible IPA phoneme string.
///
/// Pipeline:
/// 1. For English, run the hybrid G2P pipeline (phonemize_token_hybrid custom rules + espeak fallback)
/// 2. For other languages, use the pure-Rust `espeak-rs` compatibility layer
/// 3. Apply Kokoro-specific cleanup
/// 4. Filter to only characters in the Kokoro vocab
pub fn phonemize(text: &str, language: &str, vocab: &HashMap<String, u32>) -> TtsResult<String> {
    let espeak_lang = lang_to_espeak(language);

    // English spelling is highly non-phonetic and requires a G2P dictionary lookup (voice-g2p)
    // to avoid robotic espeak accents. Other languages are highly phonetic or rules-consistent,
    // so they are phonemized at the sentence level to preserve natural word linking (liaison).
    let joined = if espeak_lang == "en-us" || espeak_lang == "en-gb" {
        // Hybrid G2P: convert word-by-word to preserve OOV terms (like names or code symbols)
        let mut output = String::new();
        let mut token = String::new();
        for ch in text.chars() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '\'' | '’' | '-') {
                token.push(ch);
            } else {
                if !token.is_empty() {
                    output.push_str(&phonemize_token_hybrid(&token, espeak_lang, vocab));
                    token.clear();
                }
                output.push(ch);
            }
        }
        if !token.is_empty() {
            output.push_str(&phonemize_token_hybrid(&token, espeak_lang, vocab));
        }
        output
    } else {
        let raw_phonemes = text_to_phonemes(text, espeak_lang, None, true, false).map_err(|e| {
            TtsError::ModelError(format!(
                "pure-Rust phonemization failed for lang '{language}' (compat voice '{espeak_lang}'): {e}"
            ))
        })?;
        raw_phonemes.join("")
    };

    let replaced = apply_kokoro_replacements(&joined);
    let mut filtered = filter_to_vocab(&replaced, vocab);

    if filtered.is_empty() {
        return Err(TtsError::TokenizerError(format!(
            "Phonemization produced no valid tokens for text: \"{text}\" (lang: {language})"
        )));
    }

    // Prepend a leading space to ensure there is a tiny silent pad at the start of the audio.
    // This prevents the first word (like "I" or "A") from being clipped due to browser/OS audio device wakeup latency.
    if !filtered.starts_with(' ') {
        filtered = format!(" {filtered}");
    }

    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_vocab() -> HashMap<String, u32> {
        // Build a minimal vocab containing common IPA chars
        let chars = "$;:,.!?¡¿—…\"«»\u{201c}\u{201d} \
            ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnoprstuvwxyz\
            ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰ\
            ŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢ\
            ǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘ᵻᵊ";
        let mut vocab = HashMap::new();
        for (i, c) in chars.chars().enumerate() {
            vocab.insert(c.to_string(), i as u32);
        }
        vocab
    }

    #[test]
    fn test_language_from_voice() {
        assert_eq!(language_from_voice("af_heart"), "en");
        assert_eq!(language_from_voice("dm_speaker"), "en"); // unknown prefix
        assert_eq!(language_from_voice("jf_alpha"), "ja");
        assert_eq!(language_from_voice("ef_dora"), "es");
        assert_eq!(language_from_voice("ff_siwis"), "fr");
    }

    #[test]
    fn test_kokoro_replacements() {
        let input = "ʲrgxɬɝ";
        let output = apply_kokoro_replacements(input);
        assert_eq!(output, "jrɡklɚ");
    }

    #[test]
    fn test_phonemize_english() {
        let vocab = dummy_vocab();
        let result = phonemize("Hello world", "en", &vocab);
        assert!(result.is_ok(), "phonemize failed: {:?}", result.err());
        let ph = result.unwrap();
        assert!(!ph.is_empty(), "phonemes should not be empty");
        // Should contain IPA characters, not ASCII "Hello"
        assert!(
            ph.contains('ə') || ph.contains('ɛ') || ph.contains('ˈ') || ph.contains('l'),
            "Expected IPA phonemes, got: {ph}"
        );
    }

    #[test]
    fn test_phonemize_british_english_variant() {
        let vocab = dummy_vocab();
        let us = phonemize("schedule", "en", &vocab).expect("US phonemization should work");
        let gb = phonemize("schedule", "en-gb", &vocab).expect("British phonemization should work");

        assert!(!us.is_empty());
        assert!(!gb.is_empty());
        assert_ne!(us, gb, "expected dialect-specific phoneme output");
    }

    #[test]
    fn test_phonemize_multilingual_smoke() {
        let vocab = dummy_vocab();
        for (text, lang) in [
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr"),
            ("Guten Tag", "de"),
            ("Ciao mondo", "it"),
            ("Olá mundo", "pt"),
            ("こんにちは世界", "ja"),
            ("你好世界", "zh"),
            ("안녕하세요", "ko"),
            ("नमस्ते दुनिया", "hi"),
        ] {
            let result = phonemize(text, lang, &vocab);
            assert!(
                result.is_ok(),
                "phonemize failed for {lang}: {:?}",
                result.err()
            );
            assert!(
                !result.unwrap().is_empty(),
                "expected non-empty phonemes for {lang}"
            );
        }
    }

    #[test]
    fn test_filter_to_vocab() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 1);
        vocab.insert("b".to_string(), 2);
        assert_eq!(filter_to_vocab("abc", &vocab), "ab");
    }

    #[test]
    fn test_phonemizer_g2p_leading_stress() {
        let vocab = dummy_vocab();
        
        // Vowel-associated stress preserved, G2P mapped 'I' to 'aɪ'
        let result_i = phonemize("I have", "en", &vocab).unwrap();
        assert!(result_i.contains("ˌaɪ"), "Should preserve stress marker and map 'I' to 'aɪ': {}", result_i);

        // Consonant-associated stress stripped on consonant 'f'
        let result_ph = phonemize("phonemizer", "en", &vocab).unwrap();
        assert_eq!(result_ph, " fɑnɛmɪzɚ.");

        // "now" -> nˈaʊ
        let result_now = phonemize("now", "en", &vocab).unwrap();
        assert_eq!(result_now, " nˈaʊ");

        // "hitting" -> hˈɪtɪŋ
        let result_hitting = phonemize("hitting", "en", &vocab).unwrap();
        assert_eq!(result_hitting, " hˈɪtɪŋ");

        // "automatically" -> ˌɔtəmˈætəkᵊli
        let result_auto = phonemize("automatically", "en", &vocab).unwrap();
        assert_eq!(result_auto, " ˌɔtəmˈætəkᵊli");

        // "data" -> dˈeɪtə
        let result_data = phonemize("data", "en", &vocab).unwrap();
        assert_eq!(result_data, " dˈeɪtə");

        // "Compatibility" -> kəmpˌætəbˈɪləti
        let result_compat = phonemize("Compatibility", "en", &vocab).unwrap();
        assert_eq!(result_compat, " kəmpˌætəbˈɪləti");

        // "without" -> wɪðˈaʊt
        let result_without = phonemize("without", "en", &vocab).unwrap();
        assert_eq!(result_without, " wɪðˈaʊt");

        // "walkthrough" -> wˈɔkθɹuː
        let result_walkthrough = phonemize("walkthrough", "en", &vocab).unwrap();
        assert_eq!(result_walkthrough, " wˈɔkθɹuː");
    }
}
