pub(crate) fn normalize_text(text: &str) -> String {
    let mut output = String::new();
    let mut token = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '\'' | '’') {
            token.push(ch);
            continue;
        }

        if !token.is_empty() {
            output.push_str(&expand_token(&token));
            token.clear();
        }

        output.push(ch);
    }

    if !token.is_empty() {
        output.push_str(&expand_token(&token));
    }

    output
}

pub(crate) fn phonemize_clause(text: &str, british: bool) -> String {
    let normalized = normalize_text(text);
    let parts = split_clause_parts(&normalized);
    let mut output = String::new();

    for (index, part) in parts.iter().enumerate() {
        match part {
            ClausePart::Token(token) => {
                let next_word = next_token_after(&parts, index);
                output.push_str(&phonemize_token_with_next(token, next_word, british));
            }
            ClausePart::Separator(ch) => output.push(*ch),
        }
    }

    output
}

enum ClausePart {
    Token(String),
    Separator(char),
}

fn split_clause_parts(text: &str) -> Vec<ClausePart> {
    let mut parts = Vec::new();
    let mut token = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '\'' | '’') {
            token.push(ch);
            continue;
        }

        if !token.is_empty() {
            parts.push(ClausePart::Token(std::mem::take(&mut token)));
        }
        parts.push(ClausePart::Separator(ch));
    }

    if !token.is_empty() {
        parts.push(ClausePart::Token(token));
    }

    parts
}

fn next_token_after(parts: &[ClausePart], index: usize) -> Option<&str> {
    parts[index + 1..].iter().find_map(|part| match part {
        ClausePart::Token(token) => Some(token.as_str()),
        ClausePart::Separator(_) => None,
    })
}

fn starts_with_vowel_sound(word: &str) -> bool {
    if is_ascii_acronym_token(word) {
        return word
            .chars()
            .next()
            .is_some_and(initial_letter_starts_with_vowel_sound);
    }

    let normalized = word.replace('’', "'").to_ascii_lowercase();
    if normalized.is_empty() {
        return false;
    }

    if matches!(normalized.as_str(), "honest" | "honor" | "hour" | "heir") {
        return true;
    }

    normalized
        .chars()
        .next()
        .is_some_and(|ch| matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u'))
}

fn expand_token(token: &str) -> String {
    if token.chars().all(|ch| ch.is_ascii_digit()) {
        return token
            .chars()
            .filter_map(digit_name)
            .collect::<Vec<_>>()
            .join(" ");
    }

    token.to_string()
}

fn phonemize_token_with_next(token: &str, next_word: Option<&str>, british: bool) -> String {
    let normalized = token.replace('’', "'").to_ascii_lowercase();
    if normalized.is_empty() {
        return String::new();
    }

    if token.len() == 1 && token.chars().all(|ch| ch.is_ascii_uppercase()) {
        let letter = token.chars().next().unwrap_or('A');
        let prefers_word_form = matches!(normalized.as_str(), "a" | "i")
            && next_word.is_some()
            && !next_word.is_some_and(is_single_ascii_uppercase_token);
        if prefers_word_form {
            return lookup_word(&normalized, british)
                .unwrap_or_else(|| spell_letter(letter, british))
                .to_string();
        }

        return spell_letter(letter, british).to_string();
    }

    if is_ascii_acronym_token(token) {
        return phonemize_ascii_acronym(token, british);
    }

    if normalized == "the" && next_word.is_some_and(starts_with_vowel_sound) {
        return if british { "ðɪ" } else { "ði" }.to_string();
    }

    if let Some(phonemes) = lookup_word(&normalized, british) {
        return phonemes.to_string();
    }

    if let Some(stem) = normalized.strip_suffix("'s") {
        let base = phonemize_token_with_next(stem, None, british);
        return append_s_suffix(base);
    }

    if let Some(stem) = normalized.strip_suffix("s'") {
        let base = phonemize_token_with_next(stem, None, british);
        return append_s_suffix(base);
    }

    phonemize_fallback(&normalized, british)
}

fn phonemize_fallback(word: &str, british: bool) -> String {
    let chars: Vec<char> = word.chars().collect();
    if chars.is_empty() {
        return String::new();
    }

    let mut output = String::new();
    if estimated_syllables(word) > 1 {
        output.push('ˈ');
    }

    let mut index = 0;
    while index < chars.len() {
        if starts_with(&chars, index, "tion") {
            output.push_str("ʃən");
            index += 4;
            continue;
        }
        if starts_with(&chars, index, "sion") {
            output.push_str("ʒən");
            index += 4;
            continue;
        }
        if starts_with(&chars, index, "ture") {
            output.push_str(if british { "ʧə" } else { "ʧɚ" });
            index += 4;
            continue;
        }
        if starts_with(&chars, index, "eigh") {
            output.push_str("eɪ");
            index += 4;
            continue;
        }
        if starts_with(&chars, index, "igh") {
            output.push_str("aɪ");
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "ough") {
            output.push_str("oʊ");
            index += 4;
            continue;
        }
        if starts_with(&chars, index, "sch") {
            output.push_str("sk");
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "tch") {
            output.push('ʧ');
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "ch") {
            output.push('ʧ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "sh") {
            output.push('ʃ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "th") {
            output.push_str(if voiced_th(&chars, index) { "ð" } else { "θ" });
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ph") {
            output.push('f');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ng") {
            output.push('ŋ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ck") {
            output.push('k');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "qu") {
            output.push_str("kw");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "wh") {
            output.push('w');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "wr") {
            output.push('ɹ');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "kn") {
            output.push('n');
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ee") || starts_with(&chars, index, "ea") {
            output.push_str("iː");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "oo") {
            output.push_str("uː");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "oa") {
            output.push_str("oʊ");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ai") || starts_with(&chars, index, "ay") {
            output.push_str("eɪ");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "oi") || starts_with(&chars, index, "oy") {
            output.push_str("ɔɪ");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ow") || starts_with(&chars, index, "ou") {
            output.push_str("aʊ");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "er") && index + 2 == chars.len() {
            output.push_str(if british { "ə" } else { "ɚ" });
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "or") && index + 2 == chars.len() {
            output.push_str(if british { "ɔː" } else { "ɔɹ" });
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ar") && index + 2 == chars.len() {
            output.push_str(if british { "ɑː" } else { "ɑɹ" });
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "le")
            && index + 2 == chars.len()
            && index > 0
            && !is_vowel(chars[index - 1])
        {
            output.push_str("əl");
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "ing") && index + 3 == chars.len() {
            output.push_str("ɪŋ");
            index += 3;
            continue;
        }
        if starts_with(&chars, index, "ed") && index + 2 == chars.len() && index > 0 {
            output.push_str(past_tense_suffix(chars[index - 1]));
            index += 2;
            continue;
        }
        if starts_with(&chars, index, "es") && index + 2 == chars.len() && index > 0 {
            output.push_str(plural_suffix(&output));
            index += 2;
            continue;
        }
        if chars[index] == 's' && index + 1 == chars.len() && index > 0 {
            output.push_str(simple_s_suffix(&output));
            index += 1;
            continue;
        }

        match chars[index] {
            'a' => output.push_str(if has_magic_e(&chars, index) {
                "eɪ"
            } else {
                "æ"
            }),
            'e' => {
                if index + 1 == chars.len() {
                    index += 1;
                    continue;
                }
                output.push_str(if has_magic_e(&chars, index) {
                    "iː"
                } else {
                    "ɛ"
                });
            }
            'i' => output.push_str(if has_magic_e(&chars, index) {
                "aɪ"
            } else {
                "ɪ"
            }),
            'o' => output.push_str(if has_magic_e(&chars, index) {
                "oʊ"
            } else {
                "ɑ"
            }),
            'u' => output.push_str(if has_magic_e(&chars, index) {
                "juː"
            } else {
                "ʌ"
            }),
            'y' => {
                if index == 0 && chars.get(index + 1).copied().map(is_vowel).unwrap_or(false) {
                    output.push('j');
                } else if index + 1 == chars.len() {
                    output.push('i');
                } else {
                    output.push('ɪ');
                }
            }
            'b' => output.push('b'),
            'c' => output.push(
                if chars
                    .get(index + 1)
                    .copied()
                    .map(is_soft_vowel)
                    .unwrap_or(false)
                {
                    's'
                } else {
                    'k'
                },
            ),
            'd' => output.push('d'),
            'f' => output.push('f'),
            'g' => output.push(
                if chars
                    .get(index + 1)
                    .copied()
                    .map(is_soft_vowel)
                    .unwrap_or(false)
                {
                    'ʤ'
                } else {
                    'g'
                },
            ),
            'h' => output.push('h'),
            'j' => output.push('ʤ'),
            'k' => output.push('k'),
            'l' => output.push('l'),
            'm' => output.push('m'),
            'n' => output.push('n'),
            'p' => output.push('p'),
            'q' => output.push('k'),
            'r' => output.push('ɹ'),
            's' => output.push('s'),
            't' => output.push('t'),
            'v' => output.push('v'),
            'w' => output.push('w'),
            'x' => output.push_str("ks"),
            'z' => output.push('z'),
            '\'' => {}
            _ => output.push(chars[index]),
        }

        index += 1;
    }

    output
}

fn lookup_word<'a>(word: &str, british: bool) -> Option<&'a str>
where
    'static: 'a,
{
    if british {
        lookup_word_list(word, ENGLISH_WORDS_GB)
    } else {
        lookup_word_list(word, ENGLISH_WORDS_US)
    }
    .or_else(|| lookup_word_list(word, ENGLISH_WORDS_COMMON_OVERRIDES))
    .or_else(|| lookup_word_list(word, ENGLISH_WORDS))
}

pub(crate) fn lookup_dialect_word<'a>(word: &str, british: bool) -> Option<&'a str>
where
    'static: 'a,
{
    if british {
        lookup_word_list(word, ENGLISH_WORDS_GB)
    } else {
        lookup_word_list(word, ENGLISH_WORDS_US)
    }
    .or_else(|| lookup_word_list(word, ENGLISH_WORDS_COMMON_OVERRIDES))
}

fn lookup_word_list<'a>(word: &str, entries: &'a [(&'a str, &'a str)]) -> Option<&'a str> {
    entries
        .iter()
        .find_map(|(spelling, phonemes)| (*spelling == word).then_some(*phonemes))
}

fn is_ascii_acronym_token(token: &str) -> bool {
    let len = token.chars().count();
    (2..=6).contains(&len) && token.chars().all(|ch| ch.is_ascii_uppercase())
}

fn is_single_ascii_uppercase_token(token: &str) -> bool {
    token.chars().count() == 1 && token.chars().all(|ch| ch.is_ascii_uppercase())
}

fn initial_letter_starts_with_vowel_sound(ch: char) -> bool {
    matches!(
        ch.to_ascii_uppercase(),
        'A' | 'E' | 'F' | 'H' | 'I' | 'L' | 'M' | 'N' | 'O' | 'R' | 'S' | 'X'
    )
}

fn digit_name(ch: char) -> Option<&'static str> {
    Some(match ch {
        '0' => "zero",
        '1' => "one",
        '2' => "two",
        '3' => "three",
        '4' => "four",
        '5' => "five",
        '6' => "six",
        '7' => "seven",
        '8' => "eight",
        '9' => "nine",
        _ => return None,
    })
}

fn phonemize_ascii_acronym(token: &str, british: bool) -> String {
    let letters: Vec<char> = token.chars().collect();
    let mut output = String::new();

    for (index, ch) in letters.iter().copied().enumerate() {
        if index + 1 == letters.len() {
            output.push_str(spell_letter(ch, british));
        } else {
            output.push_str(spell_letter_secondary(ch, british));
        }
    }

    output
}

fn spell_letter(ch: char, british: bool) -> &'static str {
    match ch.to_ascii_uppercase() {
        'A' => "ˈeɪ",
        'B' => "bˈiː",
        'C' => "sˈiː",
        'D' => "dˈiː",
        'E' => "ˈiː",
        'F' => "ˈɛf",
        'G' => "ʤˈiː",
        'H' => "ˈeɪʧ",
        'I' => "ˈaɪ",
        'J' => "ʤˈeɪ",
        'K' => "kˈeɪ",
        'L' => "ˈɛl",
        'M' => "ˈɛm",
        'N' => "ˈɛn",
        'O' => "ˈoʊ",
        'P' => "pˈiː",
        'Q' => "kjˈuː",
        'R' => "ˈɑɹ",
        'S' => "ˈɛs",
        'T' => "tˈiː",
        'U' => "jˈuː",
        'V' => "vˈiː",
        'W' => "dˈʌbəljˌuː",
        'X' => "ˈɛks",
        'Y' => "wˈaɪ",
        'Z' => {
            if british {
                "zˈɛd"
            } else {
                "zˈiː"
            }
        }
        _ => "",
    }
}

fn spell_letter_secondary(ch: char, british: bool) -> &'static str {
    match ch.to_ascii_uppercase() {
        'A' => "ˌeɪ",
        'B' => "bˌiː",
        'C' => "sˌiː",
        'D' => "dˌiː",
        'E' => "ˌiː",
        'F' => "ˌɛf",
        'G' => "ʤˌiː",
        'H' => "ˌeɪʧ",
        'I' => "ˌaɪ",
        'J' => "ʤˌeɪ",
        'K' => "kˌeɪ",
        'L' => "ˌɛl",
        'M' => "ˌɛm",
        'N' => "ˌɛn",
        'O' => "ˌoʊ",
        'P' => "pˌiː",
        'Q' => "kjˌuː",
        'R' => "ˌɑɹ",
        'S' => "ˌɛs",
        'T' => "tˌiː",
        'U' => "jˌuː",
        'V' => "vˌiː",
        'W' => "dˌʌbəljˌuː",
        'X' => "ˌɛks",
        'Y' => "wˌaɪ",
        'Z' => {
            if british {
                "zˌɛd"
            } else {
                "zˌiː"
            }
        }
        _ => "",
    }
}

fn append_s_suffix(mut base: String) -> String {
    base.push_str(plural_suffix(&base));
    base
}

fn plural_suffix(base: &str) -> &'static str {
    if ends_with_sibilant(base) {
        "ɪz"
    } else if ends_with_voiceless(base) {
        "s"
    } else {
        "z"
    }
}

fn simple_s_suffix(base: &str) -> &'static str {
    if ends_with_voiceless(base) {
        "s"
    } else {
        "z"
    }
}

fn past_tense_suffix(previous: char) -> &'static str {
    match previous {
        't' | 'd' => "ɪd",
        'p' | 'k' | 'f' | 'c' | 's' | 'x' => "t",
        _ => "d",
    }
}

fn ends_with_sibilant(base: &str) -> bool {
    ["s", "z", "ʃ", "ʒ", "ʧ", "ʤ"]
        .iter()
        .any(|suffix| base.ends_with(suffix))
}

fn ends_with_voiceless(base: &str) -> bool {
    ["p", "t", "k", "f", "θ", "s", "ʃ", "ʧ", "ks"]
        .iter()
        .any(|suffix| base.ends_with(suffix))
}

fn starts_with(chars: &[char], index: usize, pattern: &str) -> bool {
    for (offset, expected) in pattern.chars().enumerate() {
        if chars.get(index + offset) != Some(&expected) {
            return false;
        }
    }
    true
}

fn has_magic_e(chars: &[char], index: usize) -> bool {
    chars.get(index + 2) == Some(&'e')
        && index + 3 == chars.len()
        && chars
            .get(index + 1)
            .copied()
            .map(|ch| !is_vowel(ch))
            .unwrap_or(false)
}

fn voiced_th(chars: &[char], index: usize) -> bool {
    index > 0
        && chars.get(index - 1).copied().map(is_vowel).unwrap_or(false)
        && chars.get(index + 2).copied().map(is_vowel).unwrap_or(false)
}

fn estimated_syllables(word: &str) -> usize {
    let chars: Vec<char> = word.chars().collect();
    let mut count = 0;
    let mut previous_was_vowel = false;

    for (index, ch) in chars.iter().copied().enumerate() {
        let current_is_vowel = is_vowel(ch);
        if current_is_vowel
            && !previous_was_vowel
            && !(ch == 'e' && index + 1 == chars.len() && count > 0 && !word.ends_with("le"))
        {
            count += 1;
        }
        previous_was_vowel = current_is_vowel;
    }

    count.max(1)
}

fn is_vowel(ch: char) -> bool {
    matches!(ch, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

fn is_soft_vowel(ch: char) -> bool {
    matches!(ch, 'e' | 'i' | 'y')
}

const ENGLISH_WORDS: &[(&str, &str)] = &[
    ("a", "ɐ"),
    ("an", "ən"),
    ("and", "ænd"),
    ("alice", "ˈælɪs"),
    ("aluminium", "ˌæljuˈmɪniəm"),
    ("apple", "ˈæpəl"),
    ("apples", "ˈæpəlz"),
    ("but", "bʌt"),
    ("caterpillar", "ˈkætəɹˌpɪlɚ"),
    ("changed", "ʧeɪnʤd"),
    ("compiler", "kəmpˈaɪlɚ"),
    ("entirely", "ɛntˈaɪɚli"),
    ("eight", "eɪt"),
    ("five", "faɪv"),
    ("for", "fɔɹ"),
    ("four", "fɔɹ"),
    ("full", "fʊl"),
    ("hardly", "ˈhɑɹdli"),
    ("have", "hæv"),
    ("hello", "həlˈoʊ"),
    ("help", "hɛlp"),
    ("i", "aɪ"),
    ("in", "ɪn"),
    ("is", "ɪz"),
    ("know", "noʊ"),
    ("kokoro", "kəkˈɔːɹoʊ"),
    ("of", "ʌv"),
    ("on", "ɑn"),
    ("one", "wˈʌn"),
    ("orange", "ˈɔɹɪnʤ"),
    ("oranges", "ˈɔɹɪnʤɪz"),
    ("opens", "ˈoʊpənz"),
    ("patient", "pˈeɪʃənt"),
    ("prototype", "ˈproʊtətaɪp"),
    ("rather", "ˈɹæðɚ"),
    ("replied", "ɹɪˈplaɪd"),
    ("running", "ɹˈʌnɪŋ"),
    ("rust", "ɹˈʌst"),
    ("said", "sɛd"),
    ("serves", "sɚvz"),
    ("seven", "ˈsɛvən"),
    ("shipped", "ʃɪpt"),
    ("shyly", "ˈʃaɪli"),
    ("six", "sɪks"),
    ("sir", "sɚ"),
    ("speech", "spˈiːʧ"),
    ("still", "stɪl"),
    ("surprisingly", "sɚpɹˈaɪzɪŋli"),
    ("team", "tˈiːm"),
    ("test", "tˈɛst"),
    ("text", "tˈɛkst"),
    ("the", "ðə"),
    ("this", "ðɪs"),
    ("three", "θɹˈiː"),
    ("thursday", "θˈɚzdeɪ"),
    ("to", "tə"),
    ("two", "tˈuː"),
    ("who", "huː"),
    ("world", "wɚld"),
    ("you", "juː"),
    ("zero", "ˈzɪɹoʊ"),
    ("nine", "naɪn"),
];

const ENGLISH_WORDS_COMMON_OVERRIDES: &[(&str, &str)] = &[
    ("walkthrough", "wˈɔkθɹuː"),
    ("walkthroughs", "wˈɔkθɹuːz"),
];

const ENGLISH_WORDS_US: &[(&str, &str)] = &[("schedule", "ˈskɛʤuːl")];

const ENGLISH_WORDS_GB: &[(&str, &str)] = &[
    ("aluminium", "ˌaljʊmˈɪniəm"),
    ("are", "ɑː"),
    ("compiler", "kəmpˈaɪlə"),
    ("for", "fɔː"),
    ("four", "fɔː"),
    ("full", "fˈʊl"),
    ("on", "ɒn"),
    ("opens", "ˈəʊpənz"),
    ("rather", "ˈɹɑːðə"),
    ("schedule", "ʃˈɛdjuːl"),
    ("serves", "sɜːvz"),
    ("shipped", "ʃˈɪpt"),
    ("sir", "sɜː"),
    ("still", "stˈɪl"),
    ("team", "tˈiːm"),
    ("thursday", "θˈɜːzdeɪ"),
    ("world", "wɜːld"),
    ("prototype", "pɹˈəʊtəʊtˌaɪp"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_text_expands_acronyms_and_digits() {
        assert_eq!(
            normalize_text("AI API serves GPU TTS on the CPU and 3 apples."),
            "AI API serves GPU TTS on the CPU and three apples."
        );
    }

    #[test]
    fn test_acronyms_are_grouped_into_single_phoneme_spans() {
        let us = phonemize_clause("The AI API serves GPU TTS on the CPU.", false);
        let gb = phonemize_clause("A CLI and an HTTP API help the GPU team.", true);

        assert!(us.contains("ˌeɪˈaɪ"), "unexpected AI output: {us}");
        assert!(us.contains("ʤˌiːpˌiːjˈuː"), "unexpected GPU output: {us}");
        assert!(gb.starts_with("ɐ "), "unexpected article output: {gb}");
        assert!(
            gb.contains("ˌeɪʧtˌiːtˌiːpˈiː"),
            "unexpected HTTP output: {gb}"
        );
    }

    #[test]
    fn test_schedule_differs_by_dialect() {
        let us = phonemize_clause("The schedule changed on Thursday.", false);
        let gb = phonemize_clause("The schedule changed on Thursday.", true);
        assert!(
            us.contains("skɛʤuːl"),
            "unexpected US schedule output: {us}"
        );
        assert!(
            gb.contains("ʃˈɛdjuːl"),
            "unexpected GB schedule output: {gb}"
        );
        assert_ne!(us, gb);
    }

    #[test]
    fn test_the_before_vowel_uses_strong_form() {
        let us = phonemize_clause("the API opens", false);
        let gb = phonemize_clause("the aluminium prototype", true);

        assert!(us.starts_with("ði"), "unexpected US article output: {us}");
        assert!(gb.starts_with("ðɪ"), "unexpected GB article output: {gb}");
    }

    #[test]
    fn test_letter_spelling_is_available() {
        assert_eq!(phonemize_clause("A I", false), "ˈeɪ ˈaɪ");
        assert_eq!(spell_letter('Z', true), "zˈɛd");
    }

    #[test]
    fn test_fallback_word_is_non_empty() {
        let phonemes = phonemize_clause("workflow", false);
        assert!(!phonemes.is_empty());
        assert!(phonemes.contains('ˈ'));
    }
}
