//! Locale-aware numeric text normalization before sentence splitting and G2P.
//!
//! Keep quantities as words so vocabulary filtering cannot silently erase them.
//! Decimal fractions and leading-zero identifiers are spoken digit by digit;
//! ambiguous groups of three digits follow the locale's grouping separator.

pub(super) fn normalize_numbers(text: &str, language: &str) -> String {
    let normalized = normalize_clock_times(ascii_digits(text), language);
    if matches!(language, "en-us" | "en-gb") {
        // The English frontend owns contextual years, ordinals, money and units.
        return normalized;
    }
    let chars: Vec<char> = normalized.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        let mut currency = None;
        if is_currency(chars[i]) {
            currency = Some(chars[i]);
            i += 1;
            while chars.get(i).is_some_and(|ch| ch.is_whitespace()) {
                i += 1;
            }
        }
        let sign = if matches!(chars.get(i), Some('-' | '−' | '+')) {
            let sign = chars[i];
            i += 1;
            Some(sign)
        } else {
            None
        };
        if !chars.get(i).is_some_and(char::is_ascii_digit) {
            output.push(chars[start]);
            i = start + 1;
            continue;
        }
        let number_start = i;
        while i < chars.len() {
            if chars[i].is_ascii_digit()
                || (matches!(chars[i], '.' | ',' | '\u{00a0}' | '\u{202f}')
                    && chars.get(i + 1).is_some_and(char::is_ascii_digit))
            {
                // Nonbreaking spaces are unambiguous thousands separators;
                // dots and commas are resolved according to the locale below.
                i += 1;
            } else {
                break;
            }
        }
        let number: String = chars[number_start..i]
            .iter()
            .filter(|ch| !matches!(ch, '\u{00a0}' | '\u{202f}'))
            .collect();
        let mut suffix = i;
        while chars.get(suffix).is_some_and(|ch| ch.is_whitespace()) {
            suffix += 1;
        }
        let percent = chars.get(suffix) == Some(&'%');
        if percent {
            i = suffix + 1;
        } else if currency.is_none() && chars.get(suffix).is_some_and(|ch| is_currency(*ch)) {
            currency = chars.get(suffix).copied();
            i = suffix + 1;
        }
        if output.chars().last().is_some_and(|ch| ch.is_alphanumeric()) {
            output.push(' ');
        }
        if let Some(sign) = sign {
            output.push_str(if sign == '+' {
                plus(language)
            } else {
                minus(language)
            });
            output.push(' ');
        }
        if percent && language == "cmn" {
            output.push_str("百分之");
        }
        let (whole, fraction) = split_number(&number, language);
        output.push_str(&integer_words(&whole, language));
        if let Some(fraction) = fraction {
            output.push(' ');
            output.push_str(decimal_word(language));
            output.push(' ');
            output.push_str(&spell_digits(&fraction, language));
        }
        if let Some(currency) = currency {
            output.push(' ');
            output.push_str(currency_word(currency, language, number == "1"));
        }
        if percent && language != "cmn" {
            output.push(' ');
            output.push_str(percent_word(language));
        }
        if chars.get(i).is_some_and(|ch| ch.is_alphanumeric()) {
            output.push(' ');
        }
    }
    output
}

/// Recognize clock fields before ':' can be interpreted as a pause or part of
/// an unknown word. Requiring two minute/second digits leaves ordinary ratios
/// such as 1:2 alone; invalid clock ranges also retain their original spelling.
fn normalize_clock_times(text: String, language: &str) -> String {
    if !text.contains(':') {
        return text;
    }
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while cursor < text.len() {
        let rest = &text[cursor..];
        let ch = rest.chars().next().unwrap();
        let word_boundary = cursor == 0
            || !text[..cursor]
                .chars()
                .next_back()
                .unwrap()
                .is_ascii_alphanumeric();
        if word_boundary && ch.is_ascii_digit() {
            if let Some((length, hour, minute, seconds)) = read_clock(rest) {
                let words = |n| {
                    if matches!(language, "en-us" | "en-gb") {
                        super::g2p::number_words(n)
                    } else {
                        cardinal(n, language)
                    }
                };
                let h = words(hour);
                let m = words(minute);
                let time = match language {
                    "en-us" | "en-gb" if minute == 0 => format!("{h} o'clock"),
                    "en-us" | "en-gb" if minute < 10 => format!("{h} oh {m}"),
                    "en-us" | "en-gb" => format!("{h} {m}"),
                    "fr" if hour == 1 => format!("une heure {m}"),
                    "fr" => format!("{h} heures {m}"),
                    "de" if hour == 1 => format!("ein Uhr {m}"),
                    "de" => format!("{h} Uhr {m}"),
                    "es" if hour == 1 => format!("una hora {m}"),
                    "pt" if hour == 1 => format!("uma hora {m}"),
                    "es" | "pt" => format!("{h} horas {m}"),
                    "it" if hour == 1 => format!("una ora {m}"),
                    "it" => format!("{h} ore {m}"),
                    "ja" => format!("{h}時{m}分"),
                    "cmn" => format!("{h}点{m}分"),
                    "ko" => format!("{}시 {m}분", korean_clock_hour(hour)),
                    "hi" => format!("{h} बजकर {m} मिनट"),
                    _ => unreachable!("canonical language"),
                };
                output.push_str(&time);
                if let Some(seconds) = seconds {
                    let unit = match language {
                        "en-us" | "en-gb" => {
                            if seconds == 1 {
                                "second"
                            } else {
                                "seconds"
                            }
                        }
                        "fr" => {
                            if seconds == 1 {
                                "seconde"
                            } else {
                                "secondes"
                            }
                        }
                        "de" => {
                            if seconds == 1 {
                                "Sekunde"
                            } else {
                                "Sekunden"
                            }
                        }
                        "es" | "pt" => {
                            if seconds == 1 {
                                "segundo"
                            } else {
                                "segundos"
                            }
                        }
                        "it" => {
                            if seconds == 1 {
                                "secondo"
                            } else {
                                "secondi"
                            }
                        }
                        "ja" | "cmn" => "秒",
                        "ko" => "초",
                        "hi" => "सेकंड",
                        _ => unreachable!("canonical language"),
                    };
                    output.push_str(&format!(" {} {unit}", words(seconds)));
                }
                cursor += length;
                if text[cursor..]
                    .chars()
                    .next()
                    .is_some_and(char::is_alphabetic)
                {
                    output.push(' ');
                }
                continue;
            }
        }
        output.push(ch);
        cursor += ch.len_utf8();
    }
    output
}

fn korean_clock_hour(hour: u64) -> &'static str {
    [
        "영",
        "한",
        "두",
        "세",
        "네",
        "다섯",
        "여섯",
        "일곱",
        "여덟",
        "아홉",
        "열",
        "열한",
        "열두",
        "열세",
        "열네",
        "열다섯",
        "열여섯",
        "열일곱",
        "열여덟",
        "열아홉",
        "스무",
        "스물한",
        "스물두",
        "스물세",
    ][hour as usize]
}

fn read_clock(text: &str) -> Option<(usize, u64, u64, Option<u64>)> {
    let bytes = text.as_bytes();
    let digits = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if !(1..=2).contains(&digits) || bytes.get(digits) != Some(&b':') {
        return None;
    }
    let field = |start| -> Option<u64> {
        let chars = bytes.get(start..start + 2)?;
        chars
            .iter()
            .all(u8::is_ascii_digit)
            .then(|| ((chars[0] - b'0') * 10 + chars[1] - b'0') as u64)
    };
    let hour = text[..digits].parse::<u64>().ok()?;
    let minute = field(digits + 1)?;
    let mut end = digits + 3;
    let seconds = if bytes.get(end) == Some(&b':') {
        let seconds = field(end + 1)?;
        end += 3;
        Some(seconds)
    } else {
        None
    };
    if hour > 23
        || minute > 59
        || seconds.is_some_and(|s| s > 59)
        || bytes.get(end).is_some_and(u8::is_ascii_digit)
    {
        return None;
    }
    Some((end, hour, minute, seconds))
}

fn ascii_digits(text: &str) -> String {
    text.chars()
        .map(|ch| {
            for base in [0x0660, 0x06f0, 0x0966, 0xff10] {
                if (base..base + 10).contains(&(ch as u32)) {
                    return char::from_u32('0' as u32 + ch as u32 - base).unwrap();
                }
            }
            ch
        })
        .collect()
}

fn split_number(number: &str, language: &str) -> (String, Option<String>) {
    let grouping = match language {
        "de" | "it" | "pt" | "es" => Some('.'),
        // French uses spaces for grouping, never its decimal comma.
        "fr" => None,
        _ => Some(','),
    };
    let separators: Vec<(usize, char)> = number
        .char_indices()
        .filter(|(_, c)| matches!(c, '.' | ','))
        .collect();
    if separators.is_empty() {
        return (number.to_string(), None);
    }
    let (_, last) = *separators.last().unwrap();
    if let Some(grouping) = grouping {
        let groups: Vec<&str> = number.split(grouping).collect();
        let international_groups =
            (1..=3).contains(&groups[0].len()) && groups.iter().skip(1).all(|part| part.len() == 3);
        let indian_groups = language == "hi"
            && groups.len() >= 2
            && (1..=2).contains(&groups[0].len())
            && groups.last().is_some_and(|part| part.len() == 3)
            && groups[1..groups.len() - 1]
                .iter()
                .all(|part| part.len() == 2);
        if separators.iter().all(|(_, c)| *c == grouping) && (international_groups || indian_groups)
        {
            return (groups.concat(), None);
        }
    }
    // When both separators occur, the last one denotes the decimal fraction.
    // Accept a single dot decimal in comma locales (e.g. scientific text 3.14).
    let decimal = number.rfind(last).unwrap();
    let whole = number[..decimal]
        .chars()
        .filter(char::is_ascii_digit)
        .collect();
    let fraction = number[decimal + 1..]
        .chars()
        .filter(char::is_ascii_digit)
        .collect();
    (whole, Some(fraction))
}

fn integer_words(number: &str, language: &str) -> String {
    if (number.len() > 1 && number.starts_with('0')) || number.len() > 15 {
        return spell_digits(number, language);
    }
    match number.parse::<u64>() {
        Ok(value) => cardinal(value, language),
        Err(_) => spell_digits(number, language),
    }
}

fn spell_digits(number: &str, language: &str) -> String {
    number
        .chars()
        .filter_map(|ch| ch.to_digit(10))
        .map(|digit| small(digit as u64, language).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

fn small(n: u64, language: &str) -> &'static str {
    let values: &[&str] = match language {
        "fr" => &[
            "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
            "onze", "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit",
            "dix-neuf",
        ],
        "de" => &[
            "null",
            "eins",
            "zwei",
            "drei",
            "vier",
            "fünf",
            "sechs",
            "sieben",
            "acht",
            "neun",
            "zehn",
            "elf",
            "zwölf",
            "dreizehn",
            "vierzehn",
            "fünfzehn",
            "sechzehn",
            "siebzehn",
            "achtzehn",
            "neunzehn",
        ],
        "es" => &[
            "cero",
            "uno",
            "dos",
            "tres",
            "cuatro",
            "cinco",
            "seis",
            "siete",
            "ocho",
            "nueve",
            "diez",
            "once",
            "doce",
            "trece",
            "catorce",
            "quince",
            "dieciséis",
            "diecisiete",
            "dieciocho",
            "diecinueve",
        ],
        "it" => &[
            "zero",
            "uno",
            "due",
            "tre",
            "quattro",
            "cinque",
            "sei",
            "sette",
            "otto",
            "nove",
            "dieci",
            "undici",
            "dodici",
            "tredici",
            "quattordici",
            "quindici",
            "sedici",
            "diciassette",
            "diciotto",
            "diciannove",
        ],
        "pt" => &[
            "zero",
            "um",
            "dois",
            "três",
            "quatro",
            "cinco",
            "seis",
            "sete",
            "oito",
            "nove",
            "dez",
            "onze",
            "doze",
            "treze",
            "catorze",
            "quinze",
            "dezesseis",
            "dezessete",
            "dezoito",
            "dezenove",
        ],
        "ja" => &["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"],
        "cmn" => &["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"],
        "ko" => &["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"],
        "hi" => HINDI_NUMBERS,
        _ => unreachable!("normalized language"),
    };
    values[n as usize]
}

fn cardinal(n: u64, language: &str) -> String {
    if matches!(language, "ja" | "cmn" | "ko") {
        return cjk_number(n, language);
    }
    if language == "hi" {
        return hindi_number(n);
    }
    if n < 20 {
        return small(n, language).into();
    }
    if n < 100 {
        return tens(n, language);
    }
    if n < 1000 {
        let h = n / 100;
        let rem = n % 100;
        let head = match language {
            "de" => format!("{}hundert", if h == 1 { "ein" } else { small(h, language) }),
            "fr" => {
                if h == 1 {
                    "cent".into()
                } else {
                    format!(
                        "{} cent{}",
                        small(h, language),
                        if rem == 0 { "s" } else { "" }
                    )
                }
            }
            "es" => [
                "",
                "ciento",
                "doscientos",
                "trescientos",
                "cuatrocientos",
                "quinientos",
                "seiscientos",
                "setecientos",
                "ochocientos",
                "novecientos",
            ][h as usize]
                .to_string(),
            "it" => format!(
                "{}cent{}",
                if h == 1 { "" } else { small(h, language) },
                if rem == 8 || (80..90).contains(&rem) {
                    ""
                } else {
                    "o"
                }
            ),
            "pt" => [
                "",
                "cento",
                "duzentos",
                "trezentos",
                "quatrocentos",
                "quinhentos",
                "seiscentos",
                "setecentos",
                "oitocentos",
                "novecentos",
            ][h as usize]
                .to_string(),
            _ => unreachable!(),
        };
        if n == 100 && language == "es" {
            return "cien".into();
        }
        if n == 100 && language == "pt" {
            return "cem".into();
        }
        if rem == 0 {
            return head;
        }
        let join = if matches!(language, "de" | "it") {
            ""
        } else if language == "pt" {
            " e "
        } else {
            " "
        };
        return format!("{head}{join}{}", cardinal(rem, language));
    }
    let scales: &[(u64, &str, &str)] = match language {
        "fr" => &[
            (1_000_000_000_000, "billion", "billions"),
            (1_000_000_000, "milliard", "milliards"),
            (1_000_000, "million", "millions"),
            (1000, "mille", "mille"),
        ],
        "de" => &[
            (1_000_000_000_000, "Billion", "Billionen"),
            (1_000_000_000, "Milliarde", "Milliarden"),
            (1_000_000, "Million", "Millionen"),
            (1000, "tausend", "tausend"),
        ],
        "es" => &[
            (1_000_000_000_000, "billón", "billones"),
            (1_000_000, "millón", "millones"),
            (1000, "mil", "mil"),
        ],
        "it" => &[
            (1_000_000_000_000, "bilione", "bilioni"),
            (1_000_000_000, "miliardo", "miliardi"),
            (1_000_000, "milione", "milioni"),
            (1000, "mille", "mila"),
        ],
        "pt" => &[
            (1_000_000_000_000, "trilhão", "trilhões"),
            (1_000_000_000, "bilhão", "bilhões"),
            (1_000_000, "milhão", "milhões"),
            (1000, "mil", "mil"),
        ],
        _ => unreachable!(),
    };
    let &(scale, singular, plural) = scales.iter().find(|(value, _, _)| n >= *value).unwrap();
    let q = n / scale;
    let rem = n % scale;
    let head = if scale == 1000 && q == 1 {
        if language == "de" {
            "eintausend".into()
        } else {
            singular.into()
        }
    } else if scale == 1000 && matches!(language, "it" | "de") {
        format!("{}{plural}", cardinal(q, language))
    } else {
        let count = if q == 1 {
            match language {
                "es" | "it" => "un".into(),
                "de" => "eine".into(),
                _ => cardinal(q, language),
            }
        } else {
            cardinal(q, language)
        };
        format!("{count} {}", if q == 1 { singular } else { plural })
    };
    if rem == 0 {
        head
    } else {
        format!(
            "{head}{}{}",
            if language == "pt" && rem < 100 {
                " e "
            } else if scale == 1000 && language == "de" {
                ""
            } else {
                " "
            },
            cardinal(rem, language)
        )
    }
}

fn tens(n: u64, language: &str) -> String {
    let t = n / 10;
    let r = n % 10;
    match language {
        "de" => {
            let base = [
                "", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig",
                "achtzig", "neunzig",
            ][t as usize];
            if r == 0 {
                base.into()
            } else {
                format!(
                    "{}und{base}",
                    if r == 1 { "ein" } else { small(r, language) }
                )
            }
        }
        "fr" => {
            if n >= 80 {
                let rem = n - 80;
                return if rem == 0 {
                    "quatre-vingts".into()
                } else {
                    format!("quatre-vingt-{}", cardinal(rem, language))
                };
            }
            if n >= 70 {
                return format!(
                    "soixante{}{}",
                    if n == 71 { " et " } else { "-" },
                    cardinal(n - 60, language)
                );
            }
            let base = [
                "",
                "",
                "vingt",
                "trente",
                "quarante",
                "cinquante",
                "soixante",
            ][t as usize];
            if r == 0 {
                base.into()
            } else {
                format!(
                    "{base}{}{}",
                    if r == 1 { " et " } else { "-" },
                    small(r, language)
                )
            }
        }
        "es" => {
            if n < 30 {
                return [
                    "veinte",
                    "veintiuno",
                    "veintidós",
                    "veintitrés",
                    "veinticuatro",
                    "veinticinco",
                    "veintiséis",
                    "veintisiete",
                    "veintiocho",
                    "veintinueve",
                ][(n - 20) as usize]
                    .into();
            }
            let base = [
                "",
                "",
                "",
                "treinta",
                "cuarenta",
                "cincuenta",
                "sesenta",
                "setenta",
                "ochenta",
                "noventa",
            ][t as usize];
            if r == 0 {
                base.into()
            } else {
                format!("{base} y {}", small(r, language))
            }
        }
        "it" => {
            let base = [
                "",
                "",
                "venti",
                "trenta",
                "quaranta",
                "cinquanta",
                "sessanta",
                "settanta",
                "ottanta",
                "novanta",
            ][t as usize];
            if r == 0 {
                base.into()
            } else {
                format!(
                    "{}{}",
                    if matches!(r, 1 | 8) {
                        &base[..base.len() - 1]
                    } else {
                        base
                    },
                    if r == 3 { "tré" } else { small(r, language) }
                )
            }
        }
        "pt" => {
            let base = [
                "",
                "",
                "vinte",
                "trinta",
                "quarenta",
                "cinquenta",
                "sessenta",
                "setenta",
                "oitenta",
                "noventa",
            ][t as usize];
            if r == 0 {
                base.into()
            } else {
                format!("{base} e {}", small(r, language))
            }
        }
        _ => unreachable!(),
    }
}

fn cjk_number(n: u64, language: &str) -> String {
    if n < 10 {
        return small(n, language).into();
    }
    let units = match language {
        "ko" => [
            (1_000_000_000_000, "조"),
            (100_000_000, "억"),
            (10000, "만"),
            (1000, "천"),
            (100, "백"),
            (10, "십"),
        ],
        "ja" => [
            (1_000_000_000_000, "兆"),
            (100_000_000, "億"),
            (10000, "万"),
            (1000, "千"),
            (100, "百"),
            (10, "十"),
        ],
        _ => [
            (1_000_000_000_000, "兆"),
            (100_000_000, "亿"),
            (10000, "万"),
            (1000, "千"),
            (100, "百"),
            (10, "十"),
        ],
    };
    let &(scale, unit) = units.iter().find(|(scale, _)| n >= *scale).unwrap();
    let q = n / scale;
    let rem = n % scale;
    let omit_one =
        q == 1 && ((language != "cmn" && scale < 10000) || (language == "cmn" && scale == 10));
    let head = format!(
        "{}{unit}",
        if omit_one {
            String::new()
        } else {
            cjk_number(q, language)
        }
    );
    if rem == 0 {
        head
    } else {
        let zero = if language == "cmn" && rem < scale / 10 {
            "零"
        } else {
            ""
        };
        // Inside larger Mandarin numbers 110 is 一百一十, not 一百十.
        let rest = if language == "cmn" && (10..20).contains(&rem) {
            format!("一{}", cjk_number(rem, language))
        } else {
            cjk_number(rem, language)
        };
        format!("{head}{zero}{rest}")
    }
}

const HINDI_NUMBERS: &[&str] = &[
    "शून्य",
    "एक",
    "दो",
    "तीन",
    "चार",
    "पाँच",
    "छह",
    "सात",
    "आठ",
    "नौ",
    "दस",
    "ग्यारह",
    "बारह",
    "तेरह",
    "चौदह",
    "पंद्रह",
    "सोलह",
    "सत्रह",
    "अठारह",
    "उन्नीस",
    "बीस",
    "इक्कीस",
    "बाईस",
    "तेईस",
    "चौबीस",
    "पच्चीस",
    "छब्बीस",
    "सत्ताईस",
    "अट्ठाईस",
    "उनतीस",
    "तीस",
    "इकतीस",
    "बत्तीस",
    "तैंतीस",
    "चौंतीस",
    "पैंतीस",
    "छत्तीस",
    "सैंतीस",
    "अड़तीस",
    "उनतालीस",
    "चालीस",
    "इकतालीस",
    "बयालीस",
    "तैंतालीस",
    "चवालीस",
    "पैंतालीस",
    "छियालीस",
    "सैंतालीस",
    "अड़तालीस",
    "उनचास",
    "पचास",
    "इक्यावन",
    "बावन",
    "तिरपन",
    "चौवन",
    "पचपन",
    "छप्पन",
    "सत्तावन",
    "अट्ठावन",
    "उनसठ",
    "साठ",
    "इकसठ",
    "बासठ",
    "तिरसठ",
    "चौंसठ",
    "पैंसठ",
    "छियासठ",
    "सड़सठ",
    "अड़सठ",
    "उनहत्तर",
    "सत्तर",
    "इकहत्तर",
    "बहत्तर",
    "तिहत्तर",
    "चौहत्तर",
    "पचहत्तर",
    "छिहत्तर",
    "सतहत्तर",
    "अठहत्तर",
    "उनासी",
    "अस्सी",
    "इक्यासी",
    "बयासी",
    "तिरासी",
    "चौरासी",
    "पचासी",
    "छियासी",
    "सत्तासी",
    "अट्ठासी",
    "नवासी",
    "नब्बे",
    "इक्यानवे",
    "बानवे",
    "तिरानवे",
    "चौरानवे",
    "पचानवे",
    "छियानवे",
    "सत्तानवे",
    "अट्ठानवे",
    "निन्यानवे",
];
fn hindi_number(n: u64) -> String {
    if n < 100 {
        return HINDI_NUMBERS[n as usize].into();
    }
    let (scale, unit) = [
        (10_000_000, "करोड़"),
        (100_000, "लाख"),
        (1000, "हज़ार"),
        (100, "सौ"),
    ]
    .into_iter()
    .find(|(scale, _)| n >= *scale)
    .unwrap();
    let head = format!("{} {unit}", hindi_number(n / scale));
    if n % scale == 0 {
        head
    } else {
        format!("{head} {}", hindi_number(n % scale))
    }
}

fn is_currency(ch: char) -> bool {
    matches!(ch, '$' | '€' | '£' | '¥' | '₹' | '₩')
}
fn currency_word(ch: char, lang: &str, singular: bool) -> &'static str {
    let (one, many) = match (lang, ch) {
        ("fr", '€') => ("euro", "euros"),
        ("fr", '$') => ("dollar", "dollars"),
        ("fr", '£') => ("livre", "livres"),
        ("de", '€') => ("Euro", "Euro"),
        ("de", '$') => ("Dollar", "Dollar"),
        ("de", '£') => ("Pfund", "Pfund"),
        ("es", '€') => ("euro", "euros"),
        ("es", '$') => ("dólar", "dólares"),
        ("es", '£') => ("libra", "libras"),
        ("it", '€') => ("euro", "euro"),
        ("it", '$') => ("dollaro", "dollari"),
        ("it", '£') => ("sterlina", "sterline"),
        ("pt", '€') => ("euro", "euros"),
        ("pt", '$') => ("dólar", "dólares"),
        ("pt", '£') => ("libra", "libras"),
        ("ja", '$') => ("ドル", "ドル"),
        ("ja", '€') => ("ユーロ", "ユーロ"),
        ("ja", '£') => ("ポンド", "ポンド"),
        ("ja", '¥') => ("円", "円"),
        ("ja", '₹') => ("ルピー", "ルピー"),
        ("ja", '₩') => ("ウォン", "ウォン"),
        ("cmn", '$') => ("美元", "美元"),
        ("cmn", '€') => ("欧元", "欧元"),
        ("cmn", '£') => ("英镑", "英镑"),
        ("cmn", '¥') => ("元", "元"),
        ("cmn", '₹') => ("卢比", "卢比"),
        ("cmn", '₩') => ("韩元", "韩元"),
        ("ko", '$') => ("달러", "달러"),
        ("ko", '€') => ("유로", "유로"),
        ("ko", '£') => ("파운드", "파운드"),
        ("ko", '¥') => ("엔", "엔"),
        ("ko", '₹') => ("루피", "루피"),
        ("ko", '₩') => ("원", "원"),
        ("hi", '$') => ("डॉलर", "डॉलर"),
        ("hi", '€') => ("यूरो", "यूरो"),
        ("hi", '£') => ("पाउंड", "पाउंड"),
        ("hi", '¥') => ("येन", "येन"),
        ("hi", '₹') => ("रुपया", "रुपये"),
        ("hi", '₩') => ("वॉन", "वॉन"),
        ("fr", '₹') => ("roupie", "roupies"),
        ("de", '₹') => ("Rupie", "Rupien"),
        ("it", '₹') => ("rupia", "rupie"),
        (_, '₹') => ("rupia", "rupias"),
        (_, '¥') => ("yen", "yen"),
        (_, '₩') => ("won", "won"),
        _ => unreachable!(),
    };
    if singular {
        one
    } else {
        many
    }
}
fn decimal_word(lang: &str) -> &'static str {
    match lang {
        "fr" => "virgule",
        "de" => "Komma",
        "es" => "coma",
        "it" => "virgola",
        "pt" => "vírgula",
        "ja" => "点",
        "cmn" => "点",
        "ko" => "점",
        "hi" => "दशमलव",
        _ => unreachable!(),
    }
}
fn percent_word(lang: &str) -> &'static str {
    match lang {
        "fr" => "pour cent",
        "de" => "Prozent",
        "es" => "por ciento",
        "it" => "per cento",
        "pt" => "por cento",
        "ja" => "パーセント",
        "cmn" => "百分比",
        "ko" => "퍼센트",
        "hi" => "प्रतिशत",
        _ => unreachable!(),
    }
}
fn minus(lang: &str) -> &'static str {
    match lang {
        "fr" => "moins",
        "de" => "minus",
        "es" => "menos",
        "it" => "meno",
        "pt" => "menos",
        "ja" => "マイナス",
        "cmn" => "负",
        "ko" => "마이너스",
        "hi" => "ऋण",
        _ => unreachable!(),
    }
}
fn plus(lang: &str) -> &'static str {
    match lang {
        "fr" => "plus",
        "de" => "plus",
        "es" => "más",
        "it" => "più",
        "pt" => "mais",
        "ja" => "プラス",
        "cmn" => "正",
        "ko" => "플러스",
        "hi" => "धन",
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn localizes_quantities_without_losing_precision() {
        for (text, lang, expected) in [
            ("J'ai 3 pommes.", "fr", "J'ai trois pommes."),
            ("3.14", "de", "drei Komma eins vier"),
            ("€12,50", "fr", "douze virgule cinq zéro euros"),
            ("1.000,25 €", "es", "mil coma dos cinco euros"),
            ("-50%", "it", "meno cinquanta per cento"),
            ("００７", "ja", "零 零 七"),
            ("३", "hi", "तीन"),
            ("1001", "cmn", "一千零一"),
            ("123", "ko", "백이십삼"),
            ("42", "pt", "quarenta e dois"),
        ] {
            assert_eq!(normalize_numbers(text, lang), expected, "{text}, {lang}");
        }
    }
    #[test]
    fn cardinal_boundaries() {
        for (n, lang, expected) in [
            (71, "fr", "soixante et onze"),
            (80, "fr", "quatre-vingts"),
            (21, "de", "einundzwanzig"),
            (100, "es", "cien"),
            (108, "it", "centotto"),
            (100, "pt", "cem"),
            (110, "cmn", "一百一十"),
            (100000, "hi", "एक लाख"),
        ] {
            assert_eq!(cardinal(n, lang), expected);
        }
    }

    #[test]
    fn clock_and_grouping_boundaries() {
        for (text, language, expected) in [
            ("10:03", "en-us", "ten oh three"),
            ("10:30", "en-gb", "ten thirty"),
            ("12:00:05", "en-us", "twelve o'clock five seconds"),
            ("12:30", "fr", "douze heures trente"),
            ("1:30", "ko", "한시 삼십분"),
            ("现在10:30", "cmn", "现在十点三十分"),
            ("1,234", "fr", "un virgule deux trois quatre"),
            ("1,23,456", "hi", "एक लाख तेईस हज़ार चार सौ छप्पन"),
            (
                "1.234,56",
                "de",
                "eintausendzweihundertvierunddreißig Komma fünf sechs",
            ),
            (
                "1\u{202f}234,56",
                "fr",
                "mille deux cent trente-quatre virgule cinq six",
            ),
        ] {
            assert_eq!(
                normalize_numbers(text, language),
                expected,
                "{text}: {language}"
            );
        }
        for invalid in ["25:30", "12:90", "1:2", "12:00:90", "12:300"] {
            assert_eq!(normalize_clock_times(invalid.into(), "en-us"), invalid);
        }
    }
    #[test]
    fn all_languages_cover_boundaries_and_large_identifiers() {
        for lang in ["fr", "de", "es", "it", "pt", "ja", "cmn", "ko", "hi"] {
            for n in [
                0u64,
                1,
                19,
                20,
                21,
                70,
                71,
                80,
                81,
                99,
                100,
                101,
                999,
                1000,
                1001,
                10000,
                1_000_000,
                1_000_000_000_000,
            ] {
                let words = normalize_numbers(&n.to_string(), lang);
                assert!(
                    !words.is_empty() && !words.chars().any(|ch| ch.is_ascii_digit()),
                    "{lang}: {n} -> {words}"
                );
            }
            assert!(!normalize_numbers("123456789012345678901234567890", lang)
                .chars()
                .any(|ch| ch.is_ascii_digit()));
        }
    }
}
