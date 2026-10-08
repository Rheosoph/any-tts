use std::error::Error;
use std::fmt;
use std::sync::OnceLock;

use super::english_g2p;
use kana2phone::kana2phone;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer;
use pinyin::ToPinyin;

pub type ESpeakResult<T> = Result<T, ESpeakError>;

#[derive(Debug, Clone)]
pub struct ESpeakError(pub String);

impl fmt::Display for ESpeakError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pure Rust eSpeak compatibility error: {}", self.0)
    }
}

impl Error for ESpeakError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SupportedLanguage {
    EnglishUs,
    EnglishGb,
    Japanese,
    Chinese,
    Korean,
    French,
    German,
    Italian,
    Portuguese,
    Spanish,
    Hindi,
}

pub fn text_to_phonemes(
    text: &str,
    language: &str,
    phoneme_separator: Option<char>,
    remove_lang_switch_flags: bool,
    remove_stress: bool,
) -> ESpeakResult<Vec<String>> {
    let canonical = canonical_language(language)
        .ok_or_else(|| ESpeakError(format!("unsupported Kokoro language '{language}'")))?;
    let language = parse_language(language).ok_or_else(|| {
        ESpeakError(format!(
            "unsupported Kokoro language '{language}' for pure-Rust espeak compatibility"
        ))
    })?;

    let normalized = super::text_normalization::normalize_numbers(text, canonical);
    let mut sentences = Vec::new();
    for line in normalized.lines() {
        for (body, terminator) in split_sentences(line) {
            let mut phonemes = phonemize_body(&body, language)?;
            if remove_lang_switch_flags {
                phonemes = strip_lang_switch_flags(&phonemes);
            }
            if remove_stress {
                phonemes = phonemes
                    .chars()
                    .filter(|ch| !matches!(ch, 'ˈ' | 'ˌ'))
                    .collect();
            }

            phonemes = normalize_output(&phonemes);
            phonemes.push(terminator);

            if let Some(separator) = phoneme_separator {
                phonemes = apply_phoneme_separator(&phonemes, separator);
            }

            sentences.push(phonemes);
        }
    }

    Ok(sentences)
}

pub(super) fn canonical_language(language: &str) -> Option<&'static str> {
    match language
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-")
        .as_str()
    {
        "en" | "en-us" | "english" | "a" => Some("en-us"),
        "en-gb" | "en-uk" | "en-gb-x-rp" | "british" | "b" => Some("en-gb"),
        "ja" | "ja-jp" | "jp" | "japanese" | "j" => Some("ja"),
        "zh" | "zh-cn" | "zh-hans" | "cmn" | "mandarin" | "z" => Some("cmn"),
        "ko" | "ko-kr" | "korean" | "k" => Some("ko"),
        "fr" | "fr-fr" | "french" | "f" => Some("fr"),
        "de" | "de-de" | "german" | "d" => Some("de"),
        "it" | "it-it" | "italian" | "i" => Some("it"),
        "pt" | "pt-br" | "pt-pt" | "portuguese" | "p" => Some("pt"),
        "es" | "es-es" | "es-mx" | "spanish" | "e" => Some("es"),
        "hi" | "hi-in" | "hindi" | "h" => Some("hi"),
        _ => None,
    }
}

fn parse_language(language: &str) -> Option<SupportedLanguage> {
    match canonical_language(language)? {
        "en" | "en-us" | "en_us" | "english" | "a" => Some(SupportedLanguage::EnglishUs),
        "en-gb" | "en_uk" | "en-gb-x-rp" | "british" | "b" => Some(SupportedLanguage::EnglishGb),
        "ja" | "jp" | "japanese" | "j" => Some(SupportedLanguage::Japanese),
        "zh" | "zh-cn" | "cmn" | "mandarin" | "z" => Some(SupportedLanguage::Chinese),
        "ko" | "korean" | "k" => Some(SupportedLanguage::Korean),
        "fr" | "french" | "f" => Some(SupportedLanguage::French),
        "de" | "german" | "d" => Some(SupportedLanguage::German),
        "it" | "italian" | "i" => Some(SupportedLanguage::Italian),
        "pt" | "pt-br" | "pt-pt" | "portuguese" | "p" => Some(SupportedLanguage::Portuguese),
        "es" | "spanish" | "e" => Some(SupportedLanguage::Spanish),
        "hi" | "hindi" | "h" => Some(SupportedLanguage::Hindi),
        _ => None,
    }
}

fn split_sentences(line: &str) -> Vec<(String, char)> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '.'
            && current.chars().last().is_some_and(|c| c.is_ascii_digit())
            && chars.peek().is_some_and(|c| c.is_ascii_digit())
        {
            current.push(ch);
            continue;
        }
        if let Some(punctuation) = normalize_punctuation(ch) {
            if matches!(punctuation, '.' | '!' | '?') {
                if !current.trim().is_empty() {
                    sentences.push((current.trim().to_string(), punctuation));
                    current.clear();
                }
            } else {
                current.push(punctuation);
            }
        } else {
            current.push(ch);
        }
    }

    if !current.trim().is_empty() {
        sentences.push((current.trim().to_string(), '.'));
    }

    sentences
}

fn phonemize_body(text: &str, language: SupportedLanguage) -> ESpeakResult<String> {
    match language {
        SupportedLanguage::EnglishUs => phonemize_english(text, false),
        SupportedLanguage::EnglishGb => phonemize_english(text, true),
        SupportedLanguage::Japanese => phonemize_by_runs(text, phonemize_japanese_run),
        SupportedLanguage::Chinese => phonemize_by_runs(text, phonemize_chinese_run),
        SupportedLanguage::Korean => phonemize_by_runs(text, phonemize_korean_run),
        SupportedLanguage::French => Ok(phonemize_french_phrase(text)),
        SupportedLanguage::German => phonemize_by_runs(text, |run| Ok(phonemize_german_word(run))),
        SupportedLanguage::Italian => {
            phonemize_by_runs(text, |run| Ok(phonemize_italian_word(run)))
        }
        SupportedLanguage::Portuguese => {
            phonemize_by_runs(text, |run| Ok(phonemize_portuguese_word(run)))
        }
        SupportedLanguage::Spanish => {
            phonemize_by_runs(text, |run| Ok(phonemize_spanish_word(run)))
        }
        SupportedLanguage::Hindi => phonemize_by_runs(text, phonemize_hindi_run),
    }
}

fn phonemize_english(text: &str, british: bool) -> ESpeakResult<String> {
    phonemize_english_clause(&normalize_inline_punctuation(text), british)
}

fn phonemize_english_clause(text: &str, british: bool) -> ESpeakResult<String> {
    Ok(super::g2p::engine(british).convert(text))
}

#[cfg(test)]
fn normalize_english_text(text: &str) -> String {
    english_g2p::normalize_text(text)
}

fn phonemize_japanese_run(text: &str) -> ESpeakResult<String> {
    let tokenizer = japanese_tokenizer()?;
    let mut tokens = tokenizer
        .tokenize(text)
        .map_err(|err| ESpeakError(format!("Japanese tokenization failed: {err}")))?;

    let mut output = String::new();
    for token in tokens.iter_mut() {
        let surface = token.surface.to_string();
        if surface.trim().is_empty() {
            if !output.ends_with(' ') {
                output.push(' ');
            }
            continue;
        }

        if is_ascii_word(&surface) {
            output.push_str(&phonemize_japanese_ascii_token(&surface));
            output.push(' ');
            continue;
        }

        let details = token.details();
        let reading = details
            .get(8)
            .copied()
            .filter(|value| *value != "*" && *value != "UNK")
            .or_else(|| {
                details
                    .get(7)
                    .copied()
                    .filter(|value| *value != "*" && *value != "UNK")
            })
            .map(str::to_string)
            .unwrap_or_else(|| to_katakana(&surface));

        let phones = kana2phone(&reading);
        let mapped = phones
            .split_whitespace()
            .map(map_japanese_phone)
            .collect::<String>();

        output.push_str(&mapped);
        output.push(' ');
    }

    Ok(output)
}

fn is_ascii_word(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|ch| ch.is_ascii_alphanumeric())
}

fn phonemize_japanese_ascii_token(token: &str) -> String {
    let lower = token.to_ascii_lowercase();
    if let Some(phonemes) = lookup_word_list(lower.as_str(), JAPANESE_ASCII_WORD_LIST) {
        return phonemes.to_string();
    }

    if token.chars().all(|ch| ch.is_ascii_uppercase()) {
        return english_g2p::phonemize_clause(token, false);
    }

    english_g2p::phonemize_clause(token, false)
}

pub(super) fn prewarm_japanese() -> ESpeakResult<()> {
    japanese_tokenizer().map(|_| ())
}

fn japanese_tokenizer() -> ESpeakResult<&'static Tokenizer> {
    // Tokenizer::tokenize takes &self and allocates a request-local lattice.
    // Sharing its immutable dictionary permits concurrent Japanese requests.
    static TOKENIZER: OnceLock<ESpeakResult<Tokenizer>> = OnceLock::new();

    let state = TOKENIZER.get_or_init(|| {
        let dictionary = load_dictionary("embedded://ipadic").map_err(|err| {
            ESpeakError(format!("failed to load embedded Lindera dictionary: {err}"))
        })?;
        let segmenter = Segmenter::new(Mode::Normal, dictionary, None);
        Ok(Tokenizer::new(segmenter))
    });

    match state {
        Ok(tokenizer) => Ok(tokenizer),
        Err(err) => Err(err.clone()),
    }
}

fn phonemize_chinese_run(text: &str) -> ESpeakResult<String> {
    let mut units = chinese_units(text);
    apply_chinese_tone_sandhi(&mut units);

    let mut output = String::new();

    for unit in units {
        match unit {
            ChineseUnit::Syllable { base, tone } => {
                output.push_str(&map_pinyin_base_and_tone(&base, tone));
            }
            ChineseUnit::Literal(text) => {
                output.push_str(&phonemize_chinese_literal(&text)?);
            }
        }
        if !output.ends_with(' ') {
            output.push(' ');
        }
    }

    Ok(output)
}

#[derive(Debug, Clone)]
enum ChineseUnit {
    Syllable { base: String, tone: u8 },
    Literal(String),
}

fn chinese_units(text: &str) -> Vec<ChineseUnit> {
    let mut units = Vec::new();
    let mut literal = String::new();

    let mut cursor = 0;
    while cursor < text.len() {
        let rest = &text[cursor..];
        if let Some((phrase, syllables)) = chinese_phrase_override(rest) {
            if !literal.is_empty() {
                units.push(ChineseUnit::Literal(std::mem::take(&mut literal)));
            }
            push_chinese_phrase_units(&mut units, syllables);
            cursor += phrase.len();
            continue;
        }

        let Some(ch) = rest.chars().next() else {
            break;
        };
        if let Some(pinyin) = ch.to_pinyin() {
            if !literal.is_empty() {
                units.push(ChineseUnit::Literal(std::mem::take(&mut literal)));
            }
            let (base, tone) = split_tone_number(pinyin.with_tone_num_end());
            units.push(ChineseUnit::Syllable { base, tone });
        } else if ch.is_ascii_alphanumeric() {
            literal.push(ch);
        } else if !literal.is_empty() {
            units.push(ChineseUnit::Literal(std::mem::take(&mut literal)));
        }

        cursor += ch.len_utf8();
    }

    if !literal.is_empty() {
        units.push(ChineseUnit::Literal(literal));
    }

    units
}

fn chinese_phrase_override(text: &str) -> Option<(&'static str, &'static [&'static str])> {
    CHINESE_PHRASE_PINYIN
        .iter()
        .filter(|(phrase, _)| text.starts_with(*phrase))
        .max_by_key(|(phrase, _)| phrase.chars().count())
        .map(|(phrase, syllables)| (*phrase, *syllables))
}

fn push_chinese_phrase_units(units: &mut Vec<ChineseUnit>, syllables: &[&str]) {
    for syllable in syllables {
        let (base, tone) = split_tone_number(syllable);
        units.push(ChineseUnit::Syllable { base, tone });
    }
}

fn phonemize_chinese_literal(text: &str) -> ESpeakResult<String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }

    phonemize_english_clause(text, false)
}

fn apply_chinese_tone_sandhi(units: &mut [ChineseUnit]) {
    let mut start = 0;

    while start < units.len() {
        while start < units.len() && matches!(units[start], ChineseUnit::Literal(_)) {
            start += 1;
        }

        let mut end = start;
        while end < units.len() && matches!(units[end], ChineseUnit::Syllable { .. }) {
            end += 1;
        }

        if start == end {
            continue;
        }

        apply_bu_yi_sandhi(&mut units[start..end]);
        apply_third_tone_sandhi(&mut units[start..end]);
        start = end;
    }
}

fn apply_bu_yi_sandhi(units: &mut [ChineseUnit]) {
    for index in 0..units.len().saturating_sub(1) {
        let next_tone = chinese_tone(&units[index + 1]).unwrap_or(5);

        let Some(base) = chinese_base(&units[index]).map(str::to_string) else {
            continue;
        };

        let Some(current_tone) = chinese_tone(&units[index]) else {
            continue;
        };

        let replacement = if base == "yi" && current_tone == 1 {
            if next_tone == 4 {
                Some(2)
            } else if next_tone != 5 {
                Some(4)
            } else {
                None
            }
        } else if base == "bu" && current_tone == 4 && next_tone == 4 {
            Some(2)
        } else {
            None
        };

        if let Some(tone) = replacement {
            set_chinese_tone(&mut units[index], tone);
        }
    }
}

fn apply_third_tone_sandhi(units: &mut [ChineseUnit]) {
    let mut index = 0;

    while index < units.len() {
        if chinese_tone(&units[index]) != Some(3) {
            index += 1;
            continue;
        }

        let mut end = index + 1;
        while end < units.len() && chinese_tone(&units[end]) == Some(3) {
            end += 1;
        }

        if end - index > 1 {
            for syllable in &mut units[index..end - 1] {
                set_chinese_tone(syllable, 2);
            }
        }

        index = end;
    }
}

fn chinese_base(unit: &ChineseUnit) -> Option<&str> {
    match unit {
        ChineseUnit::Syllable { base, .. } => Some(base.as_str()),
        ChineseUnit::Literal(_) => None,
    }
}

fn chinese_tone(unit: &ChineseUnit) -> Option<u8> {
    match unit {
        ChineseUnit::Syllable { tone, .. } => Some(*tone),
        ChineseUnit::Literal(_) => None,
    }
}

fn set_chinese_tone(unit: &mut ChineseUnit, tone: u8) {
    if let ChineseUnit::Syllable { tone: current, .. } = unit {
        *current = tone;
    }
}

fn phonemize_korean_run(text: &str) -> ESpeakResult<String> {
    let mut output = String::new();
    let mut ascii = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            ascii.push(ch);
            continue;
        }

        if !ascii.is_empty() {
            output.push_str(&phonemize_english_clause(&ascii, false)?);
            ascii.clear();
        }

        if let Some((onset, vowel, coda)) = decompose_hangul_syllable(ch) {
            output.push_str(KOREAN_ONSETS[onset]);
            output.push_str(KOREAN_VOWELS[vowel]);
            output.push_str(KOREAN_CODAS[coda]);
        }
    }

    if !ascii.is_empty() {
        output.push_str(&phonemize_english_clause(&ascii, false)?);
    }

    Ok(output)
}

fn phonemize_hindi_run(text: &str) -> ESpeakResult<String> {
    if let Some(exception) = lookup_word_list(text, HINDI_WORD_LIST) {
        return Ok(exception.to_string());
    }

    if is_ascii_word(text) {
        return phonemize_english_clause(text, false);
    }

    let chars: Vec<char> = text.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        let ch = chars[index];

        if let Some(vowel) = hindi_independent_vowel(ch) {
            output.push_str(vowel);
            index += 1;
            continue;
        }

        if let Some(consonant) = hindi_consonant(ch) {
            output.push_str(consonant);

            let mut consumed = 0usize;
            let mut vowel = "ə";
            if let Some(next) = chars.get(index + 1).copied() {
                if next == '़' {
                    consumed += 1;
                }

                if let Some(marker) = chars.get(index + 1 + consumed).copied() {
                    if let Some(matra) = hindi_matra(marker) {
                        vowel = matra;
                        consumed += 1;
                    } else if marker == '्' {
                        vowel = "";
                        consumed += 1;
                    }
                }
            }

            let is_word_end = index + consumed + 1 >= chars.len();
            if is_word_end && vowel == "ə" {
                vowel = "";
            }

            output.push_str(vowel);
            index += consumed + 1;
            continue;
        }

        match ch {
            'ं' => output.push('ŋ'),
            'ँ' => output.push('\u{0303}'),
            'ः' => output.push('h'),
            _ if ch.is_ascii_alphanumeric() => {
                let mut ascii = String::new();
                ascii.push(ch);
                index += 1;
                while let Some(next) = chars.get(index).copied() {
                    if !next.is_ascii_alphanumeric() {
                        break;
                    }
                    ascii.push(next);
                    index += 1;
                }
                output.push_str(&phonemize_english_clause(&ascii, false)?);
                continue;
            }
            _ => {}
        }

        index += 1;
    }

    Ok(output)
}

fn phonemize_by_runs<F>(text: &str, mut phonemize_run: F) -> ESpeakResult<String>
where
    F: FnMut(&str) -> ESpeakResult<String>,
{
    let mut output = String::new();
    let mut current = String::new();

    for ch in text.chars() {
        if is_run_char(ch) {
            current.push(ch);
            continue;
        }

        if !current.is_empty() {
            output.push_str(&phonemize_run(&current)?);
            current.clear();
        }

        if ch.is_whitespace() {
            output.push(' ');
        } else if let Some(punctuation) = normalize_punctuation(ch) {
            output.push(punctuation);
        }
    }

    if !current.is_empty() {
        output.push_str(&phonemize_run(&current)?);
    }

    Ok(output)
}

fn is_run_char(ch: char) -> bool {
    !ch.is_whitespace() && normalize_punctuation(ch).is_none()
}

fn normalize_punctuation(ch: char) -> Option<char> {
    match ch {
        '.' | '。' | '।' | '॥' => Some('.'),
        ',' | '，' | '、' => Some(','),
        '¡' => Some('!'),
        '!' | '！' => Some('!'),
        '¿' => Some('?'),
        '?' | '？' => Some('?'),
        ';' | '；' => Some(';'),
        ':' | '：' => Some(':'),
        '"' | '“' | '”' | '「' | '」' | '『' | '』' => Some('"'),
        '(' | '（' => Some('('),
        ')' | '）' => Some(')'),
        '—' | '–' => Some('—'),
        '…' => Some('…'),
        _ => None,
    }
}

fn normalize_inline_punctuation(text: &str) -> String {
    text.chars()
        .map(|ch| normalize_punctuation(ch).unwrap_or(ch))
        .collect()
}

fn normalize_output(text: &str) -> String {
    let mut normalized = collapse_spaces(text);
    for punctuation in [",", ".", "!", "?", ";", ":", ")", "”"] {
        normalized = normalized.replace(&format!(" {punctuation}"), punctuation);
    }
    normalized = normalized.replace("( ", "(");
    normalized = normalized.replace("“ ", "“");
    normalized.trim().to_string()
}

#[derive(Debug, Clone)]
enum PhraseToken {
    Run(String),
    Space,
    Separator(char),
}

fn split_phrase_tokens(text: &str) -> Vec<PhraseToken> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if is_run_char(ch) {
            current.push(ch);
            continue;
        }

        if !current.is_empty() {
            tokens.push(PhraseToken::Run(std::mem::take(&mut current)));
        }

        if ch.is_whitespace() {
            if !matches!(tokens.last(), Some(PhraseToken::Space)) {
                tokens.push(PhraseToken::Space);
            }
        } else if let Some(punctuation) = normalize_punctuation(ch) {
            tokens.push(PhraseToken::Separator(punctuation));
        }
    }

    if !current.is_empty() {
        tokens.push(PhraseToken::Run(current));
    }

    tokens
}

fn next_phrase_run(tokens: &[PhraseToken], index: usize) -> Option<&str> {
    let mut cursor = index + 1;
    let mut saw_space = false;

    while let Some(token) = tokens.get(cursor) {
        match token {
            PhraseToken::Space => {
                saw_space = true;
            }
            PhraseToken::Separator(_) => return None,
            PhraseToken::Run(run) => return saw_space.then_some(run.as_str()),
        }
        cursor += 1;
    }

    None
}

fn collapse_spaces(text: &str) -> String {
    let mut output = String::new();
    let mut last_space = false;

    for ch in text.chars() {
        if ch.is_whitespace() {
            if !last_space && !output.is_empty() {
                output.push(' ');
            }
            last_space = true;
        } else {
            output.push(ch);
            last_space = false;
        }
    }

    output.trim().to_string()
}

fn apply_phoneme_separator(text: &str, separator: char) -> String {
    let units = split_phoneme_units(text);
    let mut output = String::new();
    let mut previous_was_phoneme = false;

    for unit in units {
        let is_punctuation = unit
            .chars()
            .all(|ch| ch == ' ' || normalize_punctuation(ch).is_some());
        if is_punctuation {
            output.push_str(&unit);
            previous_was_phoneme = false;
        } else {
            if previous_was_phoneme {
                output.push(separator);
            }
            output.push_str(&unit);
            previous_was_phoneme = true;
        }
    }

    output
}

fn split_phoneme_units(text: &str) -> Vec<String> {
    let mut units = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch == ' ' || normalize_punctuation(ch).is_some() {
            if !current.is_empty() {
                units.push(current.clone());
                current.clear();
            }
            units.push(ch.to_string());
            continue;
        }

        if matches!(ch, 'ˈ' | 'ˌ') {
            if !current.is_empty() {
                units.push(current.clone());
                current.clear();
            }
            current.push(ch);
            continue;
        }

        if matches!(ch, '\u{0303}' | 'ː' | 'ʰ' | 'ʲ') {
            current.push(ch);
            continue;
        }

        if current.starts_with('ˈ') || current.starts_with('ˌ') {
            current.push(ch);
            units.push(current.clone());
            current.clear();
            continue;
        }

        if !current.is_empty() {
            units.push(current.clone());
            current.clear();
        }

        current.push(ch);
    }

    if !current.is_empty() {
        units.push(current);
    }

    units
}

fn strip_lang_switch_flags(text: &str) -> String {
    let mut output = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '(' {
            output.push(ch);
            continue;
        }

        let mut flag = String::new();
        let mut valid = true;
        for next in chars.by_ref() {
            if next == ')' {
                break;
            }

            flag.push(next);
            if !next.is_ascii_alphabetic() && next != '-' {
                valid = false;
            }
        }

        if !valid || flag.is_empty() {
            output.push('(');
            output.push_str(&flag);
            output.push(')');
        }
    }

    output
}

fn phonemize_french_phrase(text: &str) -> String {
    let tokens = split_phrase_tokens(text);
    let mut output = String::new();

    for (index, token) in tokens.iter().enumerate() {
        match token {
            PhraseToken::Run(run) => {
                let lower = run.to_lowercase();
                let mut phonemes = phonemize_french_word(run);

                if let Some(next_word) = next_phrase_run(&tokens, index) {
                    if let Some(liaison) = french_liaison_sound(&lower, next_word) {
                        phonemes.push_str(liaison);
                    }
                }

                output.push_str(&phonemes);
            }
            PhraseToken::Space => output.push(' '),
            PhraseToken::Separator(ch) => output.push(*ch),
        }
    }

    output
}

fn french_liaison_sound(current_word: &str, next_word: &str) -> Option<&'static str> {
    if !french_starts_with_vowel_sound(next_word) {
        return None;
    }

    match current_word {
        "est" | "petit" | "grand" => Some("t"),
        "êtes" => Some("z"),
        "les" | "des" | "mes" | "tes" | "ses" | "nos" | "vos" | "deux" | "trois" | "sont"
        | "vous" | "nous" => Some("z"),
        "un" | "mon" | "ton" | "son" | "bon" => Some("n"),
        "leurs" => Some("ʁ"),
        _ => None,
    }
}

fn french_starts_with_vowel_sound(word: &str) -> bool {
    word.chars().next().is_some_and(|ch| {
        matches!(
            ch,
            'a' | 'e'
                | 'h'
                | 'i'
                | 'o'
                | 'u'
                | 'y'
                | 'à'
                | 'â'
                | 'ä'
                | 'é'
                | 'è'
                | 'ê'
                | 'ë'
                | 'î'
                | 'ï'
                | 'ô'
                | 'ö'
                | 'ù'
                | 'û'
                | 'ü'
                | 'œ'
        )
    })
}

fn to_katakana(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ('ぁ'..='ゖ').contains(&ch) {
                char::from_u32(ch as u32 + 0x60).unwrap_or(ch)
            } else {
                ch
            }
        })
        .collect()
}

fn map_japanese_phone(phone: &str) -> String {
    if let Some(base) = phone.strip_suffix(':') {
        return format!("{}ː", map_japanese_phone(base));
    }

    match phone {
        "a" => "a".into(),
        "i" => "i".into(),
        "u" => "ɯ".into(),
        "e" => "e".into(),
        "o" => "o".into(),
        "N" => "ɴ".into(),
        "q" => "q".into(),
        "ch" => "ʧ".into(),
        "sh" => "ʃ".into(),
        "ts" => "ʦ".into(),
        "j" => "ʥ".into(),
        "r" => "ɾ".into(),
        "ry" => "ɾj".into(),
        "ny" => "ɲ".into(),
        "hy" => "ç".into(),
        "by" => "bj".into(),
        "py" => "pj".into(),
        "my" => "mj".into(),
        "gy" => "gj".into(),
        "ky" => "kj".into(),
        "f" => "ɸ".into(),
        "y" => "j".into(),
        _ => phone.to_string(),
    }
}

fn map_pinyin_base_and_tone(base: &str, tone: u8) -> String {
    let base = base.replace('ü', "v").replace("u:", "v");

    let syllable = match base.as_str() {
        "zhi" => "ʧɻ".to_string(),
        "chi" => "ʧʰɻ".to_string(),
        "shi" => "ʃɻ".to_string(),
        "ri" => "ɻ".to_string(),
        "zi" => "ʦz".to_string(),
        "ci" => "ʦʰz".to_string(),
        "si" => "sz".to_string(),
        "yi" => "i".to_string(),
        "ya" => "ja".to_string(),
        "yan" => "jɛn".to_string(),
        "yang" => "jɑŋ".to_string(),
        "yao" => "jau".to_string(),
        "ye" => "jɛ".to_string(),
        "yin" => "in".to_string(),
        "ying" => "iŋ".to_string(),
        "yong" => "jʊŋ".to_string(),
        "you" => "jou".to_string(),
        "yu" => "y".to_string(),
        "yue" => "yɛ".to_string(),
        "yuan" => "yɛn".to_string(),
        "yun" => "yn".to_string(),
        "wu" => "u".to_string(),
        "wa" => "wa".to_string(),
        "wai" => "wai".to_string(),
        "wan" => "wan".to_string(),
        "wang" => "wɑŋ".to_string(),
        "wei" => "wei".to_string(),
        "wen" => "wən".to_string(),
        "weng" => "wəŋ".to_string(),
        "wo" => "wo".to_string(),
        _ => {
            let (initial, final_part) = split_pinyin_initial(&base);
            let onset = map_pinyin_initial(initial);
            let rhyme = map_pinyin_final(final_part);
            format!("{onset}{rhyme}")
        }
    };

    format!("{syllable}{}", chinese_tone_marker(tone))
}

fn split_tone_number(pinyin: &str) -> (String, u8) {
    match pinyin.chars().last() {
        Some(ch) if ch.is_ascii_digit() => {
            let tone = ch.to_digit(10).unwrap_or(5) as u8;
            (
                pinyin[..pinyin.len() - ch.len_utf8()].to_ascii_lowercase(),
                tone,
            )
        }
        _ => (pinyin.to_ascii_lowercase(), 5),
    }
}

fn split_pinyin_initial(pinyin: &str) -> (&str, &str) {
    for initial in [
        "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x",
        "r", "z", "c", "s",
    ] {
        if let Some(rest) = pinyin.strip_prefix(initial) {
            return (initial, rest);
        }
    }

    ("", pinyin)
}

fn map_pinyin_initial(initial: &str) -> &'static str {
    match initial {
        "b" => "p",
        "p" => "pʰ",
        "m" => "m",
        "f" => "f",
        "d" => "t",
        "t" => "tʰ",
        "n" => "n",
        "l" => "l",
        "g" => "k",
        "k" => "kʰ",
        "h" => "χ",
        "j" => "ʨ",
        "q" => "ʨʰ",
        "x" => "ɕ",
        "zh" => "ʧ",
        "ch" => "ʧʰ",
        "sh" => "ʃ",
        "r" => "ɻ",
        "z" => "ʦ",
        "c" => "ʦʰ",
        "s" => "s",
        _ => "",
    }
}

fn map_pinyin_final(final_part: &str) -> String {
    match final_part {
        "a" => "a".into(),
        "ai" => "ai".into(),
        "an" => "an".into(),
        "ang" => "ɑŋ".into(),
        "ao" => "au".into(),
        "e" => "ɤ".into(),
        "ei" => "ei".into(),
        "en" => "ən".into(),
        "eng" => "əŋ".into(),
        "er" => "ɚ".into(),
        "i" => "i".into(),
        "ia" => "ja".into(),
        "ian" => "jɛn".into(),
        "iang" => "jɑŋ".into(),
        "iao" => "jau".into(),
        "ie" => "jɛ".into(),
        "in" => "in".into(),
        "ing" => "iŋ".into(),
        "iong" => "jʊŋ".into(),
        "iu" => "jou".into(),
        "o" => "o".into(),
        "ong" => "ʊŋ".into(),
        "ou" => "ou".into(),
        "u" => "u".into(),
        "ua" => "wa".into(),
        "uai" => "wai".into(),
        "uan" => "wan".into(),
        "uang" => "wɑŋ".into(),
        "ui" => "wei".into(),
        "un" => "wən".into(),
        "uo" => "wo".into(),
        "v" => "y".into(),
        "ve" => "yɛ".into(),
        "van" => "yɛn".into(),
        "vn" => "yn".into(),
        _ => final_part.to_string(),
    }
}

fn chinese_tone_marker(tone: u8) -> &'static str {
    match tone {
        1 => "→",
        2 => "↗",
        3 => "↓",
        4 => "↘",
        _ => "",
    }
}

const S_BASE: u32 = 0xAC00;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

const KOREAN_ONSETS: [&str; 19] = [
    "k", "k", "n", "t", "t", "ɾ", "m", "p", "p", "s", "s", "", "ʧ", "ʧ", "ʧʰ", "kʰ", "tʰ", "pʰ",
    "h",
];

const KOREAN_VOWELS: [&str; 21] = [
    "a", "ɛ", "ja", "jɛ", "ʌ", "e", "jʌ", "je", "o", "wa", "wɛ", "we", "jo", "u", "wʌ", "we", "wi",
    "ju", "ɯ", "ɰi", "i",
];

const KOREAN_CODAS: [&str; 28] = [
    "", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "p", "l", "l", "p", "l", "m", "p", "p",
    "t", "t", "ŋ", "t", "t", "k", "t", "p", "t",
];

const CHINESE_PHRASE_PINYIN: &[(&str, &[&str])] = &[
    ("重庆", &["chong2", "qing4"]),
    ("银行", &["yin2", "hang2"]),
    ("音乐", &["yin1", "yue4"]),
    ("重要", &["zhong4", "yao4"]),
    ("长大", &["zhang3", "da4"]),
    ("长城", &["chang2", "cheng2"]),
];

const GERMAN_WORD_LIST: &[(&str, &str)] = &[
    ("hallo", "hˈaloː"),
    ("dies", "diːs"),
    ("ist", "ɪst"),
    ("in", "ɪn"),
    ("ein", "aɪn"),
    ("der", "dɛɾ"),
    ("und", "ʊnt"),
    ("gut", "ɡˈuːt"),
    ("bleiben", "blˈaɪbən"),
    ("bleibt", "blˈaɪpt"),
    ("sprach", "ʃpʁaːχ"),
    ("überraschend", "ˌyːbɜrˈaʃənt"),
    ("test", "tˈɛst"),
    ("schön", "ʃøːn"),
    ("kokoro", "koːkˈoːroː"),
    ("rust", "rˈʊst"),
    ("synthese", "zyntˈeːzə"),
    ("sport", "ʃpɔʁt"),
    ("vollständig", "fˈɔlʃtˌɛndɪç"),
    ("implementiert", "ˌɪmpleːməntˈiːɾt"),
];

const JAPANESE_ASCII_WORD_LIST: &[(&str, &str)] = &[("kokoro", "kokoɾo"), ("rust", "ɾasɯto")];

const FRENCH_WORD_LIST: &[(&str, &str)] = &[
    ("ami", "amˈi"),
    ("bonjour", "bɔ̃ʒˈuʁ"),
    ("ceci", "səsˌi"),
    ("est", "ɛ"),
    ("un", "œ̃"),
    ("test", "tˈɛst"),
    ("de", "də"),
    ("la", "la"),
    ("kokoro", "kokoʁˈo"),
    ("rust", "ɹˈʌst"),
    ("vous", "vu"),
    ("nous", "nu"),
    ("êtes", "ɛt"),
    ("très", "tʁɛ"),
    ("synthèse", "sɛ̃tˈɛz"),
    ("vocale", "vokˈal"),
    ("entièrement", "ɑ̃tjɛʁmˈɑ̃"),
    ("en", "ɑ̃"),
];

const SPANISH_WORD_LIST: &[(&str, &str)] = &[
    ("clara", "klˈaɾa"),
    ("de", "ðe"),
    ("es", "ˈes"),
    ("esta", "ˈesta"),
    ("hola", "ˈola"),
    ("implementada", "ˌimplementˈaða"),
    ("kokoro", "kokˈoɾo"),
    ("prueba", "pɾuˈeβa"),
    ("quiero", "kjˈeɾo"),
    ("rust", "rˈust"),
    ("síntesis", "sˈintesis"),
    ("una", "ˈuna"),
    ("voz", "βˈoθ"),
    ("yo", "ʝˈo"),
];

const ITALIAN_WORD_LIST: &[(&str, &str)] = &[
    ("chiara", "kjˈaɾa"),
    ("ciao", "ʧˈao"),
    ("della", "dˌɛlla"),
    ("interamente", "interamˈente"),
    ("italiana", "italiˈana"),
    ("kokoro", "kokˈɔro"),
    ("questo", "kwˈesto"),
    ("resta", "rˈɛsta"),
    ("rust", "rˈust"),
    ("sciarpa", "ʃˈarpa"),
    ("sintesi", "sˈintezi"),
    ("test", "tˈɛst"),
    ("un", "ʊn"),
    ("vocale", "vokˈale"),
    ("è", "ˈɛ"),
];

const PORTUGUESE_WORD_LIST: &[(&str, &str)] = &[
    ("a", "ɐ"),
    ("casa", "kˈazɐ"),
    ("da", "dɐ"),
    ("de", "dɨ"),
    ("do", "dʊ"),
    ("é", "ɛ"),
    ("em", "ẽ"),
    ("essa", "ˈɛsɐ"),
    ("este", "ˈɛʃtɨ"),
    ("fala", "fˈalɐ"),
    ("implementada", "ˌimplementˈadɐ"),
    ("inteiramente", "ĩteɾamˈẽtɨ"),
    ("kokoro", "kokˈɔɾo"),
    ("mar", "mˈaɹ"),
    ("não", "nˈɐ̃w"),
    ("olá", "ɔlˈa"),
    ("rust", "ʁˈuʃt"),
    ("síntese", "sˈiŋtɨzɨ"),
    ("teste", "tˈɛʃtɨ"),
    ("um", "ũŋ"),
    ("usa", "ˈuzɐ"),
];

const HINDI_WORD_LIST: &[(&str, &str)] = &[
    ("का", "kaː"),
    ("कर", "kˈʌɾ"),
    ("गए", "ɡˈʌeː"),
    ("ध्वनि", "dʰʋˈʌnɪ"),
    ("नमस्ते", "nəmˈʌsteː"),
    ("परीक्षण", "pəɾˈiːkʃəɳ"),
    ("पाठ-से-भाषण", "pˈaːʈʰseːbʰˈaːʂəɳ"),
    ("में", "mẽː"),
    ("यह", "jˌəh"),
    ("रहे", "ɾˌəheː"),
    ("लिखे", "lˈɪkʰeː"),
    ("संश्लेषण", "sənʃlˈeːʂəɳ"),
    ("हम", "hˌəm"),
    ("है", "hɛː"),
    ("हैं", "hɛ̃"),
];

const GERMAN_COMPOUND_TAILS: &[&str] = &["synthese", "sport", "werk", "art", "frau"];

const GERMAN_PREFIXES: &[(&str, &str)] = &[
    ("über", "yːbɐ"),
    ("ver", "fɛɐ"),
    ("ent", "ɛnt"),
    ("zer", "ʦɛɐ"),
    ("miss", "mɪs"),
];

const GERMAN_SUFFIXES: &[(&str, &str)] = &[
    ("schaft", "ʃaft"),
    ("keit", "kaɪt"),
    ("heit", "haɪt"),
    ("lich", "lɪç"),
    ("ung", "ʊŋ"),
    ("nis", "nɪs"),
];

fn lookup_word_list<'a>(word: &str, entries: &'a [(&'a str, &'a str)]) -> Option<&'a str> {
    entries
        .iter()
        .find_map(|(spelling, phonemes)| (*spelling == word).then_some(*phonemes))
}

fn lookup_latin_exception(
    word: &str,
    entries: &'static [(&'static str, &'static str)],
) -> Option<&'static str> {
    lookup_word_list(word, entries)
}

fn decompose_hangul_syllable(ch: char) -> Option<(usize, usize, usize)> {
    let code = ch as u32;
    if !(S_BASE..S_BASE + S_COUNT).contains(&code) {
        return None;
    }

    let s_index = code - S_BASE;
    let onset = (s_index / N_COUNT) as usize;
    let vowel = ((s_index % N_COUNT) / T_COUNT) as usize;
    let coda = (s_index % T_COUNT) as usize;
    Some((onset, vowel, coda))
}

fn phonemize_spanish_word(word: &str) -> String {
    let lower = word.to_lowercase();
    if let Some(exception) = lookup_latin_exception(&lower, SPANISH_WORD_LIST) {
        return exception.to_string();
    }

    let chars: Vec<char> = lower.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        if starts_with(&chars, index, "gue") || starts_with(&chars, index, "gui") {
            output.push('g');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ch") {
            output.push('ʧ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ll") {
            output.push('ʝ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "rr") {
            output.push('r');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "qu") {
            output.push('k');
            index += 2;
            continue;
        }

        let ch = chars[index];
        let previous = if index == 0 {
            None
        } else {
            chars.get(index - 1).copied()
        };
        let next = chars.get(index + 1).copied();
        match ch {
            'a' => output.push('a'),
            'á' => push_stressed_phoneme(&mut output, "a"),
            'e' => output.push('e'),
            'é' => push_stressed_phoneme(&mut output, "e"),
            'i' => output.push('i'),
            'í' => push_stressed_phoneme(&mut output, "i"),
            'o' => output.push('o'),
            'ó' => push_stressed_phoneme(&mut output, "o"),
            'u' | 'ü' => output.push('u'),
            'ú' => push_stressed_phoneme(&mut output, "u"),
            'b' | 'v' => output.push(if is_word_final(index, &chars) {
                'p'
            } else {
                'b'
            }),
            'c' => {
                if matches!(next, Some('e' | 'é' | 'i' | 'í')) {
                    output.push('s');
                } else {
                    output.push('k');
                }
            }
            'd' => output.push(if is_word_final(index, &chars) {
                't'
            } else {
                'd'
            }),
            'f' => output.push('f'),
            'g' => {
                if matches!(next, Some('e' | 'é' | 'i' | 'í')) {
                    output.push('χ');
                } else {
                    output.push(if is_word_final(index, &chars) {
                        'k'
                    } else {
                        'g'
                    });
                }
            }
            'h' => {}
            'j' => output.push('χ'),
            'k' => output.push('k'),
            'l' => output.push('l'),
            'm' => output.push('m'),
            'n' => output.push('n'),
            'ñ' => output.push('ɲ'),
            'p' => output.push('p'),
            'q' => output.push('k'),
            'r' => {
                let previous = if index == 0 {
                    None
                } else {
                    chars.get(index - 1).copied()
                };
                if index == 0 || previous.map(|value| !is_vowel(value)).unwrap_or(false) {
                    output.push('r');
                } else {
                    output.push('ɾ');
                }
            }
            's' => output.push('s'),
            't' => output.push('t'),
            'w' => output.push('w'),
            'x' => output.push_str("ks"),
            'y' => {
                if is_word_final(index, &chars) {
                    output.push('i');
                } else if previous.map(is_vowel).unwrap_or(false)
                    || next.map(is_vowel).unwrap_or(false)
                {
                    output.push('j');
                } else {
                    output.push('i');
                }
            }
            'z' => output.push('s'),
            '\'' | '’' => {}
            _ => output.push(ch),
        }

        index += 1;
    }

    output
}

fn phonemize_italian_word(word: &str) -> String {
    let lower = word.to_lowercase();
    if let Some(exception) = lookup_latin_exception(&lower, ITALIAN_WORD_LIST) {
        return exception.to_string();
    }

    let chars: Vec<char> = lower.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        if starts_with(&chars, index, "qu") {
            output.push_str("kw");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "gli") {
            output.push('ʎ');
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "gn") {
            output.push('ɲ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ch") {
            output.push('k');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "gh") {
            output.push('g');
            index += 2;
            continue;
        }
        if chars.get(index + 1) == Some(&chars[index])
            && matches!(chars[index], 'l' | 'm' | 'n' | 'p' | 'r' | 's' | 't' | 'z')
        {
            let consonant = match chars[index] {
                'r' => 'r',
                'z' => 'ʦ',
                other => other,
            };
            output.push(consonant);
            output.push('ː');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "sci")
            && chars.get(index + 3).copied().map(is_vowel).unwrap_or(false)
        {
            output.push('ʃ');
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "sc") {
            let next = chars.get(index + 2).copied();
            if matches!(next, Some('e' | 'i')) {
                output.push('ʃ');
                index += 2;
                continue;
            }
        }

        let ch = chars[index];
        let next = chars.get(index + 1).copied();
        match ch {
            'a' => output.push('a'),
            'à' => push_stressed_phoneme(&mut output, "a"),
            'e' => output.push('e'),
            'è' | 'é' => push_stressed_phoneme(&mut output, "e"),
            'i' => output.push('i'),
            'ì' | 'í' => push_stressed_phoneme(&mut output, "i"),
            'o' => output.push('o'),
            'ò' | 'ó' => push_stressed_phoneme(&mut output, "o"),
            'u' => output.push('u'),
            'ù' | 'ú' => push_stressed_phoneme(&mut output, "u"),
            'c' => {
                if matches!(next, Some('e' | 'i')) {
                    output.push('ʧ');
                } else {
                    output.push('k');
                }
            }
            'g' => {
                if matches!(next, Some('e' | 'i')) {
                    output.push('ʤ');
                } else {
                    output.push('g');
                }
            }
            'h' => {}
            'j' => output.push('j'),
            'q' => output.push('k'),
            'r' => {
                let previous = if index == 0 {
                    None
                } else {
                    chars.get(index - 1).copied()
                };
                if index == 0 || previous.map(|value| !is_vowel(value)).unwrap_or(false) {
                    output.push('r');
                } else {
                    output.push('ɾ');
                }
            }
            's' => output.push('s'),
            't' => output.push('t'),
            'v' => output.push('v'),
            'z' => output.push('ʦ'),
            '\'' | '’' => {}
            _ => output.push(ch),
        }

        index += 1;
    }

    output
}

fn phonemize_portuguese_word(word: &str) -> String {
    let lower = word.to_lowercase();
    if let Some(exception) = lookup_latin_exception(&lower, PORTUGUESE_WORD_LIST) {
        return exception.to_string();
    }

    let chars: Vec<char> = lower.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        if starts_with(&chars, index, "ão") && index + 2 == chars.len() {
            push_stressed_phoneme(&mut output, "ɐ");
            output.push('\u{0303}');
            output.push('w');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "nh") {
            output.push('ɲ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "lh") {
            output.push('ʎ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ch") {
            output.push('ʃ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "rr") {
            output.push('ʁ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ss") {
            output.push('s');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "qu") {
            output.push('k');
            index += 2;
            continue;
        }

        let ch = chars[index];
        let previous = if index == 0 {
            None
        } else {
            chars.get(index - 1).copied()
        };
        let next = chars.get(index + 1).copied();
        if matches!(
            ch,
            'a' | 'e' | 'i' | 'o' | 'u' | 'á' | 'â' | 'ã' | 'é' | 'ê' | 'í' | 'ó' | 'ô' | 'õ' | 'ú'
        ) && matches!(next, Some('m' | 'n'))
            && chars
                .get(index + 2)
                .map(|ch| !is_vowel(*ch))
                .unwrap_or(true)
        {
            output.push_str(match ch {
                'a' | 'á' | 'â' => "a",
                'ã' => "ɐ",
                'e' | 'é' | 'ê' => "e",
                'i' | 'í' => "i",
                'o' | 'ó' | 'ô' | 'õ' => "o",
                _ => "u",
            });
            output.push('\u{0303}');
            index += 2;
            continue;
        }

        match ch {
            'a' | 'â' => output.push('a'),
            'á' => push_stressed_phoneme(&mut output, "a"),
            'ã' => {
                push_stressed_phoneme(&mut output, "ɐ");
                output.push('\u{0303}');
            }
            'e' => output.push('e'),
            'é' | 'ê' => push_stressed_phoneme(&mut output, "e"),
            'i' => output.push('i'),
            'í' => push_stressed_phoneme(&mut output, "i"),
            'o' => output.push('o'),
            'ó' | 'ô' => push_stressed_phoneme(&mut output, "o"),
            'õ' => {
                push_stressed_phoneme(&mut output, "o");
                output.push('\u{0303}');
            }
            'u' => output.push('u'),
            'ú' => push_stressed_phoneme(&mut output, "u"),
            'b' => output.push('b'),
            'c' => {
                if matches!(next, Some('e' | 'é' | 'ê' | 'i' | 'í')) {
                    output.push('s');
                } else {
                    output.push('k');
                }
            }
            'k' | 'q' => output.push('k'),
            'ç' => output.push('s'),
            'd' => output.push('d'),
            'f' => output.push('f'),
            'g' => {
                if matches!(next, Some('e' | 'é' | 'ê' | 'i' | 'í')) {
                    output.push('ʒ');
                } else {
                    output.push('g');
                }
            }
            'h' => {}
            'j' => output.push('ʒ'),
            'l' => output.push('l'),
            'm' => output.push('m'),
            'n' => output.push('n'),
            'p' => output.push('p'),
            'r' => {
                if is_word_final(index, &chars)
                    || next.map(|value| !is_vowel(value)).unwrap_or(false)
                {
                    output.push('ʁ');
                } else {
                    output.push('ɾ');
                }
            }
            's' => {
                if previous.map(is_vowel).unwrap_or(false) && next.map(is_vowel).unwrap_or(false) {
                    output.push('z');
                } else if is_word_final(index, &chars) && previous.map(is_vowel).unwrap_or(false) {
                    output.push('ʃ');
                } else {
                    output.push('s');
                }
            }
            't' => output.push('t'),
            'v' => output.push('v'),
            'x' => output.push('ʃ'),
            'z' => output.push(if is_word_final(index, &chars) {
                'ʃ'
            } else {
                'z'
            }),
            '\'' | '’' => {}
            _ => output.push(ch),
        }

        index += 1;
    }

    output
}

fn phonemize_german_word(word: &str) -> String {
    let lower = word.to_lowercase();
    if let Some(exception) = lookup_latin_exception(&lower, GERMAN_WORD_LIST) {
        return exception.to_string();
    }

    if let Some(phonemes) = phonemize_german_morphemes(&lower) {
        return phonemes;
    }

    let chars: Vec<char> = lower.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        if starts_with(&chars, index, "sp") && german_should_hush_s(&chars, index) {
            output.push_str("ʃp");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "st") && german_should_hush_s(&chars, index) {
            output.push_str("ʃt");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "tsch") {
            output.push('ʧ');
            index += 4;
            continue;
        }
        if starts_with(&chars, index, "sch") {
            output.push('ʃ');
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "pf") {
            output.push_str("pf");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ng") {
            output.push('ŋ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "nk") {
            output.push('ŋ');
            output.push('k');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "au") {
            output.push('a');
            output.push('ʊ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ei") {
            output.push('a');
            output.push('ɪ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ie") {
            output.push('i');
            output.push('ː');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "eu") || starts_with(&chars, index, "äu") {
            output.push('ɔ');
            output.push('ʏ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ph") {
            output.push('f');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "qu") {
            output.push_str("kv");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ig") && index + 2 == chars.len() {
            output.push('ɪ');
            output.push('ç');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "er") && index > 0 && index + 2 == chars.len() {
            output.push('ɐ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "en") && index > 0 && index + 2 == chars.len() {
            output.push('ə');
            output.push('n');
            index += 2;
            continue;
        }

        let ch = chars[index];
        let next = chars.get(index + 1).copied();

        if next == Some(ch)
            && matches!(
                ch,
                'b' | 'd' | 'f' | 'g' | 'k' | 'l' | 'm' | 'n' | 'p' | 'r' | 's' | 't'
            )
        {
            match ch {
                'b' => output.push('b'),
                'd' => output.push('d'),
                'f' => output.push('f'),
                'g' => output.push('g'),
                'k' => output.push('k'),
                'l' => output.push('l'),
                'm' => output.push('m'),
                'n' => output.push('n'),
                'p' => output.push('p'),
                'r' => output.push('ʁ'),
                's' => output.push('s'),
                't' => output.push('t'),
                _ => {}
            }
            index += 2;
            continue;
        }

        match ch {
            'a' => {
                if next == Some('h') {
                    output.push('a');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('a');
                if german_has_open_syllable(&chars, index) {
                    output.push('ː');
                }
            }
            'ä' => {
                if next == Some('h') {
                    output.push('ɛ');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('ɛ');
            }
            'e' => {
                if index + 1 == chars.len() {
                    output.push('ə');
                } else if next == Some('h') {
                    output.push('e');
                    output.push('ː');
                    index += 2;
                    continue;
                } else {
                    output.push('e');
                    if german_has_open_syllable(&chars, index) {
                        output.push('ː');
                    }
                }
            }
            'i' => {
                if next == Some('h') {
                    output.push('i');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('i');
                if german_has_open_syllable(&chars, index) {
                    output.push('ː');
                }
            }
            'o' => {
                if next == Some('h') {
                    output.push('o');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('o');
                if german_has_open_syllable(&chars, index) {
                    output.push('ː');
                }
            }
            'ö' => {
                if next == Some('h') {
                    output.push('ø');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('ø');
            }
            'u' => {
                if next == Some('h') {
                    output.push('u');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('u');
                if german_has_open_syllable(&chars, index) {
                    output.push('ː');
                }
            }
            'ü' => {
                if next == Some('h') {
                    output.push('y');
                    output.push('ː');
                    index += 2;
                    continue;
                }
                output.push('y');
            }
            'ß' => output.push('s'),
            'b' => output.push(if index + 1 == chars.len() { 'p' } else { 'b' }),
            'c' => {
                if matches!(next, Some('e' | 'i' | 'y' | 'ä' | 'ö' | 'ü')) {
                    output.push('ʦ');
                } else {
                    output.push('k');
                }
            }
            'd' => output.push(if index + 1 == chars.len() { 't' } else { 'd' }),
            'f' => output.push('f'),
            'g' => output.push(if index + 1 == chars.len() { 'k' } else { 'g' }),
            'h' => {
                let previous = if index == 0 {
                    None
                } else {
                    chars.get(index - 1).copied()
                };
                if index == 0 || previous.map(|value| !is_vowel(value)).unwrap_or(true) {
                    output.push('h');
                }
            }
            'j' => output.push('j'),
            'k' => output.push('k'),
            'l' => output.push('l'),
            'm' => output.push('m'),
            'n' => output.push('n'),
            'p' => output.push('p'),
            'q' => output.push('k'),
            'r' => output.push('ʁ'),
            's' => {
                let previous = if index == 0 {
                    None
                } else {
                    chars.get(index - 1).copied()
                };
                if previous.map(is_vowel).unwrap_or(false) && next.map(is_vowel).unwrap_or(false) {
                    output.push('z');
                } else {
                    output.push('s');
                }
            }
            't' => output.push('t'),
            'v' => output.push('f'),
            'w' => output.push('v'),
            'x' => output.push_str("ks"),
            'y' => output.push('y'),
            'z' => output.push('ʦ'),
            _ => output.push(ch),
        }

        if ch == 'c' && matches!(next, Some('h')) {
            output.pop();
            output.push(german_ch_sound(&chars, index));
            index += 2;
            continue;
        }

        index += 1;
    }

    output
}

fn phonemize_french_word(word: &str) -> String {
    let lower = word.to_lowercase();
    if let Some(exception) = lookup_latin_exception(&lower, FRENCH_WORD_LIST) {
        return exception.to_string();
    }

    let chars: Vec<char> = lower.chars().collect();
    let mut output = String::new();
    let mut index = 0;

    while index < chars.len() {
        let mut matched = false;
        for (pattern, replacement) in [
            ("eaux", "o"),
            ("eau", "o"),
            ("ain", "ɛ\u{0303}"),
            ("ein", "ɛ\u{0303}"),
            ("oin", "wɛ\u{0303}"),
            ("ion", "jɔ\u{0303}"),
            ("ill", "j"),
            ("ou", "u"),
            ("oi", "wa"),
            ("eu", "ø"),
            ("œu", "ø"),
            ("an", "ɑ\u{0303}"),
            ("am", "ɑ\u{0303}"),
            ("en", "ɑ\u{0303}"),
            ("em", "ɑ\u{0303}"),
            ("on", "ɔ\u{0303}"),
            ("om", "ɔ\u{0303}"),
            ("un", "œ\u{0303}"),
            ("um", "œ\u{0303}"),
            ("in", "ɛ\u{0303}"),
            ("im", "ɛ\u{0303}"),
            ("yn", "ɛ\u{0303}"),
            ("ym", "ɛ\u{0303}"),
            ("gn", "ɲ"),
            ("ch", "ʃ"),
            ("ph", "f"),
            ("qu", "k"),
        ] {
            if starts_with(&chars, index, pattern) {
                output.push_str(replacement);
                index += pattern.chars().count();
                matched = true;
                break;
            }
        }

        if matched {
            continue;
        }

        let ch = chars[index];
        let next = chars.get(index + 1).copied();
        match ch {
            'a' | 'à' | 'â' => output.push('a'),
            'e' | 'é' | 'è' | 'ê' | 'ë' => output.push('e'),
            'i' | 'î' | 'ï' => output.push('i'),
            'o' | 'ô' => output.push('o'),
            'u' | 'ù' | 'û' | 'ü' => output.push('y'),
            'b' => output.push('b'),
            'c' => {
                if matches!(next, Some('e' | 'é' | 'i' | 'y')) {
                    output.push('s');
                } else {
                    output.push('k');
                }
            }
            'd' => output.push('d'),
            'f' => output.push('f'),
            'g' => {
                if matches!(next, Some('e' | 'é' | 'i' | 'y')) {
                    output.push('ʒ');
                } else {
                    output.push('g');
                }
            }
            'h' => {}
            'j' => output.push('ʒ'),
            'k' => output.push('k'),
            'l' => output.push('l'),
            'm' => output.push('m'),
            'n' => output.push('n'),
            'p' => output.push('p'),
            'q' => output.push('k'),
            'r' => output.push('ʁ'),
            's' => output.push('s'),
            't' => output.push('t'),
            'v' => output.push('v'),
            'w' => output.push('w'),
            'x' => output.push_str("ks"),
            'y' => output.push('j'),
            'z' => output.push('z'),
            '\'' | '’' => {}
            _ => output.push(ch),
        }

        index += 1;
    }

    trim_french_silent_final(&lower, output)
}

fn trim_french_silent_final(word: &str, mut phonemes: String) -> String {
    if matches!(
        word,
        "de" | "le" | "me" | "se" | "ce" | "je" | "que" | "très" | "plus" | "tous"
    ) {
        return phonemes;
    }

    for suffix in ['e', 's', 't', 'd', 'x', 'p'] {
        if word.ends_with(suffix) {
            phonemes.pop();
            break;
        }
    }

    phonemes
}

fn german_should_hush_s(chars: &[char], index: usize) -> bool {
    if !matches!(chars.get(index + 1), Some('p' | 't')) {
        return false;
    }

    if index == 0 {
        return true;
    }

    chars
        .get(index - 1)
        .copied()
        .map(|value| !is_vowel(value) && value != 's')
        .unwrap_or(false)
}

fn german_ch_sound(chars: &[char], index: usize) -> char {
    let previous = if index == 0 {
        None
    } else {
        chars.get(index - 1).copied()
    };
    let before_previous = if index < 2 {
        None
    } else {
        chars.get(index - 2).copied()
    };

    if matches!((before_previous, previous), (Some('a'), Some('u'))) {
        return 'χ';
    }

    if previous.map(is_front_vowel).unwrap_or(false)
        || matches!(previous, Some('l' | 'n' | 'r'))
        || matches!(
            (before_previous, previous),
            (Some('e'), Some('i')) | (Some('e'), Some('u')) | (Some('ä'), Some('u'))
        )
    {
        'ç'
    } else {
        'χ'
    }
}

fn starts_with(chars: &[char], index: usize, pattern: &str) -> bool {
    for (offset, expected) in pattern.chars().enumerate() {
        if chars.get(index + offset) != Some(&expected) {
            return false;
        }
    }
    true
}

fn is_word_final(index: usize, chars: &[char]) -> bool {
    index + 1 == chars.len()
}

fn push_stressed_phoneme(output: &mut String, phoneme: &str) {
    if !output.contains('ˈ') {
        output.push('ˈ');
    }
    output.push_str(phoneme);
}

fn german_has_open_syllable(chars: &[char], index: usize) -> bool {
    let Some(next) = chars.get(index + 1).copied() else {
        return false;
    };
    let Some(after_next) = chars.get(index + 2).copied() else {
        return false;
    };

    !is_vowel(next) && is_vowel(after_next) && after_next != next
}

fn phonemize_german_morphemes(word: &str) -> Option<String> {
    if let Some((stem, suffix)) = german_split_suffix(word) {
        return Some(format!("{}{}", phonemize_german_word(stem), suffix));
    }

    if let Some((prefix, stem)) = german_split_prefix(word) {
        let stem_phonemes = phonemize_german_word(stem);
        let stem_phonemes = if stem_phonemes.contains('ˈ') {
            stem_phonemes
        } else {
            format!("ˈ{stem_phonemes}")
        };
        return Some(format!("{prefix}{stem_phonemes}"));
    }

    if let Some((left, right)) = german_split_compound(word) {
        return Some(format!(
            "{}{}",
            phonemize_german_word(left),
            phonemize_german_word(right)
        ));
    }

    None
}

fn german_split_prefix(word: &str) -> Option<(&'static str, &str)> {
    GERMAN_PREFIXES.iter().find_map(|(prefix, phonemes)| {
        word.strip_prefix(prefix)
            .filter(|stem| stem.chars().count() >= 3)
            .map(|stem| (*phonemes, stem))
    })
}

fn german_split_suffix(word: &str) -> Option<(&str, &'static str)> {
    GERMAN_SUFFIXES.iter().find_map(|(suffix, phonemes)| {
        word.strip_suffix(suffix)
            .filter(|stem| stem.chars().count() >= 2)
            .map(|stem| (stem, *phonemes))
    })
}

fn german_split_compound(word: &str) -> Option<(&str, &str)> {
    GERMAN_COMPOUND_TAILS.iter().find_map(|tail| {
        word.find(tail)
            .filter(|index| *index >= 3)
            .map(|index| (&word[..index], &word[index..]))
    })
}

fn is_vowel(ch: char) -> bool {
    matches!(
        ch,
        'a' | 'e'
            | 'i'
            | 'o'
            | 'u'
            | 'y'
            | 'á'
            | 'à'
            | 'â'
            | 'ã'
            | 'ä'
            | 'é'
            | 'è'
            | 'ê'
            | 'ë'
            | 'í'
            | 'ì'
            | 'î'
            | 'ï'
            | 'ó'
            | 'ò'
            | 'ô'
            | 'õ'
            | 'ö'
            | 'ú'
            | 'ù'
            | 'û'
            | 'ü'
            | 'ă'
    )
}

fn is_front_vowel(ch: char) -> bool {
    matches!(ch, 'e' | 'i' | 'ä' | 'ö' | 'ü' | 'é' | 'è' | 'ê' | 'y')
}

fn hindi_independent_vowel(ch: char) -> Option<&'static str> {
    Some(match ch {
        'अ' => "ə",
        'आ' => "a",
        'इ' => "i",
        'ई' => "i",
        'उ' => "u",
        'ऊ' => "u",
        'ए' => "e",
        'ऐ' => "ɛ",
        'ओ' => "o",
        'औ' => "ɔ",
        'ऋ' => "ɾi",
        _ => return None,
    })
}

fn hindi_matra(ch: char) -> Option<&'static str> {
    Some(match ch {
        'ा' => "a",
        'ि' | 'ी' => "i",
        'ु' | 'ू' => "u",
        'े' => "e",
        'ै' => "ɛ",
        'ो' => "o",
        'ौ' => "ɔ",
        'ृ' => "ɾi",
        _ => return None,
    })
}

fn hindi_consonant(ch: char) -> Option<&'static str> {
    Some(match ch {
        'क' => "k",
        'ख' => "kʰ",
        'ग' => "g",
        'घ' => "gʰ",
        'ङ' => "ŋ",
        'च' => "ʧ",
        'छ' => "ʧʰ",
        'ज' => "ʤ",
        'झ' => "ʤʰ",
        'ञ' => "ɲ",
        'ट' => "ʈ",
        'ठ' => "ʈʰ",
        'ड' => "ɖ",
        'ढ' => "ɖʰ",
        'ण' => "ɳ",
        'त' => "t",
        'थ' => "tʰ",
        'द' => "d",
        'ध' => "dʰ",
        'न' => "n",
        'प' => "p",
        'फ' => "pʰ",
        'ब' => "b",
        'भ' => "bʰ",
        'म' => "m",
        'य' => "j",
        'र' => "ɾ",
        'ल' => "l",
        'व' => "ʋ",
        'श' => "ʃ",
        'ष' => "ʂ",
        'स' => "s",
        'ह' => "h",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT_ALICE: &str =
        "Who are you? said the Caterpillar. Replied Alice, rather shyly, I hardly know, sir!";

    #[test]
    fn test_basic_en() {
        let phonemes = text_to_phonemes("test", "en-US", None, false, false)
            .expect("english phonemization should work")
            .join("");
        assert!(phonemes.ends_with('.'));
        assert!(phonemes.contains('t'));
    }

    #[test]
    fn test_empty_input_returns_no_sentences() {
        let phonemes = text_to_phonemes("   \n  ", "en-US", None, false, false)
            .expect("whitespace input should not fail");
        assert!(phonemes.is_empty());
    }

    #[test]
    fn test_unsupported_language_errors() {
        let err = text_to_phonemes("salve", "la", None, false, false)
            .expect_err("unsupported languages should error");
        assert!(err.to_string().contains("unsupported"));
    }

    #[test]
    fn test_english_acronym_normalization() {
        assert_eq!(
            normalize_english_text("AI API serves GPU TTS on the CPU."),
            "AI API serves GPU TTS on the CPU."
        );
    }

    #[test]
    fn test_it_splits_sentences() {
        let phonemes = text_to_phonemes(TEXT_ALICE, "en-US", None, false, false)
            .expect("english phonemization should work");
        assert_eq!(phonemes.len(), 3);
    }

    #[test]
    fn test_it_adds_phoneme_separator() {
        let phonemes = text_to_phonemes("test", "en-US", Some('_'), false, false)
            .expect("english phonemization should work")
            .join("");
        assert!(phonemes.contains('_'));
        assert!(phonemes.ends_with('.'));
    }

    #[test]
    fn test_it_preserves_clause_breakers() {
        let phonemes = text_to_phonemes(TEXT_ALICE, "en-US", None, false, false)
            .expect("english phonemization should work")
            .join("");
        for punctuation in [',', '.', '?', '!'] {
            assert!(
                phonemes.contains(punctuation),
                "missing punctuation {punctuation}"
            );
        }
    }

    #[test]
    fn test_stress_toggle() {
        let with_stress = text_to_phonemes(TEXT_ALICE, "en-US", None, false, false)
            .expect("english phonemization should work")
            .join("");
        let without_stress = text_to_phonemes(TEXT_ALICE, "en-US", None, false, true)
            .expect("english phonemization should work")
            .join("");

        assert!(with_stress.contains('ˈ') || with_stress.contains('ˌ'));
        assert!(!without_stress.contains('ˈ') && !without_stress.contains('ˌ'));
    }

    #[test]
    fn test_line_splitting() {
        let phonemes = text_to_phonemes("Hello\nThere\nAnd\nWelcome", "en-US", None, false, false)
            .expect("english phonemization should work");
        assert_eq!(phonemes.len(), 4);
    }

    #[test]
    fn test_kokoro_languages_smoke() {
        let cases = [
            ("Hello world", "en"),
            ("The schedule changed", "en-gb"),
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr"),
            ("Guten Tag", "de"),
            ("Ciao mondo", "it"),
            ("Olá mundo", "pt"),
            ("こんにちは世界", "ja"),
            ("你好世界", "zh"),
            ("안녕하세요", "ko"),
            ("नमस्ते दुनिया", "hi"),
        ];

        for (text, lang) in cases {
            let phonemes = text_to_phonemes(text, lang, None, false, false)
                .unwrap_or_else(|err| panic!("{lang} phonemization failed: {err}"))
                .join("");
            assert!(
                !phonemes.is_empty(),
                "expected non-empty phonemes for language {lang}"
            );
        }
    }

    #[test]
    fn test_german_word_list_and_suffix_rules() {
        assert_eq!(phonemize_german_word("hallo"), "hˈaloː");
        assert_eq!(phonemize_german_word("vollständig"), "fˈɔlʃtˌɛndɪç");
        assert_eq!(phonemize_german_word("gut"), "ɡˈuːt");
        assert_eq!(phonemize_german_word("und"), "ʊnt");
        assert_eq!(phonemize_german_word("rust"), "rˈʊst");
        assert_eq!(phonemize_german_word("kokoro"), "koːkˈoːroː");
        assert_eq!(phonemize_german_word("sprach"), "ʃpʁaːχ");
        assert_eq!(phonemize_german_word("sprachsynthese"), "ʃpʁaːχzyntˈeːzə");
        assert_eq!(phonemize_german_word("wegen"), "veːgən");
        assert_eq!(phonemize_german_word("rad"), "ʁat");
        assert_eq!(phonemize_german_word("ich"), "iç");
        assert_eq!(phonemize_german_word("bach"), "baχ");
        assert_eq!(phonemize_german_word("schönheit"), "ʃøːnhaɪt");

        let sentence = text_to_phonemes("Hallo vollständig", "de", None, false, false)
            .expect("german phonemization should work")
            .join("");
        assert!(sentence.starts_with("hˈaloː"));
        assert!(sentence.contains("ɪç"));
    }

    #[test]
    fn test_german_morpheme_and_compound_rules() {
        assert_eq!(phonemize_german_word("radsport"), "ʁatʃpɔʁt");
        assert_eq!(phonemize_german_word("überraschung"), "yːbɐˈʁaʃʊŋ");
        assert_eq!(phonemize_german_word("freundlichkeit"), "fʁɔʏntlɪçkaɪt");
        assert_eq!(phonemize_german_word("überraschend"), "ˌyːbɜrˈaʃənt");
        assert_eq!(phonemize_german_word("bleiben"), "blˈaɪbən");
        assert_eq!(phonemize_german_word("bleibt"), "blˈaɪpt");
        assert_eq!(phonemize_german_word("implementiert"), "ˌɪmpleːməntˈiːɾt");
    }

    #[test]
    fn test_japanese_ascii_loanwords_are_transliterated() {
        let phonemes = text_to_phonemes(
            "RustでKokoroの音声合成を試します。",
            "ja",
            None,
            false,
            false,
        )
        .expect("japanese phonemization should work")
        .join("");

        assert!(
            phonemes.contains("ɾasɯto"),
            "unexpected Rust output: {phonemes}"
        );
        assert!(
            phonemes.contains("kokoɾo"),
            "unexpected Kokoro output: {phonemes}"
        );
        assert!(
            !phonemes.contains("Rust"),
            "raw ASCII leaked into output: {phonemes}"
        );
    }

    #[test]
    fn test_french_word_list_common_words() {
        assert_eq!(phonemize_french_word("ami"), "amˈi");
        assert_eq!(phonemize_french_word("bonjour"), "bɔ̃ʒˈuʁ");
        assert_eq!(phonemize_french_word("de"), "də");
        assert_eq!(phonemize_french_word("synthèse"), "sɛ̃tˈɛz");
        assert_eq!(phonemize_french_word("vous"), "vu");
        assert_eq!(phonemize_french_word("êtes"), "ɛt");
    }

    #[test]
    fn test_french_liaison_common_case() {
        let phonemes = text_to_phonemes("Ceci est un ami", "fr", None, false, false)
            .expect("french phonemization should work")
            .join("");
        assert!(
            phonemes.contains("ɛt œ̃"),
            "expected est-un liaison, got {phonemes}"
        );
    }

    #[test]
    fn test_french_vous_liaison() {
        let phonemes = text_to_phonemes("Vous êtes un ami", "fr", None, false, false)
            .expect("french phonemization should work")
            .join("");
        assert!(
            phonemes.contains("vuz ɛtz"),
            "expected vous-etes liaison, got {phonemes}"
        );
        assert!(
            phonemes.contains("œ̃n amˈi"),
            "expected un-ami liaison, got {phonemes}"
        );
    }

    #[test]
    fn test_spanish_r_distinction() {
        assert_eq!(phonemize_spanish_word("rosa"), "rosa");
        assert_eq!(phonemize_spanish_word("perro"), "pero");
        assert_eq!(phonemize_spanish_word("yo"), "ʝˈo");
        assert_eq!(phonemize_spanish_word("quiero"), "kjˈeɾo");
        assert_eq!(phonemize_spanish_word("voz"), "βˈoθ");
        assert!(phonemize_spanish_word("madrid").ends_with('t'));
    }

    #[test]
    fn test_italian_word_list_and_qu_cluster() {
        assert_eq!(phonemize_italian_word("questo"), "kwˈesto");
        assert_eq!(phonemize_italian_word("sintesi"), "sˈintezi");
        assert_eq!(phonemize_italian_word("sciarpa"), "ʃˈarpa");
        assert_eq!(phonemize_italian_word("chiara"), "kjˈaɾa");
    }

    #[test]
    fn test_portuguese_word_list() {
        assert_eq!(phonemize_portuguese_word("olá"), "ɔlˈa");
        assert_eq!(phonemize_portuguese_word("síntese"), "sˈiŋtɨzɨ");
        assert_eq!(phonemize_portuguese_word("a"), "ɐ");
        assert_eq!(phonemize_portuguese_word("casa"), "kˈazɐ");
        assert_eq!(phonemize_portuguese_word("não"), "nˈɐ̃w");
        assert_eq!(phonemize_portuguese_word("do"), "dʊ");
        assert_eq!(phonemize_portuguese_word("mar"), "mˈaɹ");
        assert_eq!(phonemize_portuguese_word("um"), "ũŋ");
    }

    #[test]
    fn test_korean_ascii_segments_are_phonemized() {
        let phonemes = text_to_phonemes("Rust로 Kokoro 테스트", "ko", None, false, false)
            .expect("korean phonemization should work")
            .join("");
        assert!(
            phonemes.contains("ɹˈʌst"),
            "unexpected Rust output: {phonemes}"
        );
        assert!(
            phonemes.contains("kəkˈɔɹO"),
            "unexpected Kokoro output: {phonemes}"
        );
    }

    #[test]
    fn test_hindi_word_list_and_ascii_segments() {
        let phonemes = text_to_phonemes("नमस्ते Rust पाठ-से-भाषण", "hi", None, false, false)
            .expect("hindi phonemization should work")
            .join("");
        assert!(
            phonemes.contains("nəmˈʌsteː"),
            "unexpected नमस्ते output: {phonemes}"
        );
        assert!(
            phonemes.contains("ɹˈʌst"),
            "unexpected Rust output: {phonemes}"
        );
        assert!(
            phonemes.contains("pˈaːʈʰseːbʰˈaːʂəɳ"),
            "unexpected phrase output: {phonemes}"
        );
    }

    #[test]
    fn test_chinese_third_tone_sandhi() {
        let phonemes = text_to_phonemes("你好", "zh", None, false, false)
            .expect("chinese phonemization should work")
            .join("");
        assert!(
            phonemes.contains("ni↗"),
            "expected third-tone sandhi on 你, got {phonemes}"
        );
        assert!(
            phonemes.contains("χau↓"),
            "expected third tone on 好, got {phonemes}"
        );
    }

    #[test]
    fn test_chinese_phrase_overrides_polyphones() {
        let phonemes = text_to_phonemes("重庆银行", "zh", None, false, false)
            .expect("chinese phrase overrides should work")
            .join("");
        assert!(
            phonemes.contains("ʧʰʊŋ↗ ʨʰiŋ↘"),
            "expected chongqing override, got {phonemes}"
        );
        assert!(
            phonemes.contains("in↗ χɑŋ↗"),
            "expected yinhang override, got {phonemes}"
        );
    }

    #[test]
    fn test_chinese_ascii_loanwords_are_phonemized() {
        let phonemes = text_to_phonemes("AI语音TTS", "zh", None, false, false)
            .expect("mixed Chinese ASCII phonemization should work")
            .join("");
        assert!(
            !phonemes.contains("AI"),
            "expected ASCII AI to be phonemized, got {phonemes}"
        );
        assert!(
            !phonemes.contains("TTS"),
            "expected ASCII TTS to be phonemized, got {phonemes}"
        );
    }
}
