//! Sentence-aware English G2P adapted from voice-g2p 0.2.2.
//! The engine is entirely in process. Dictionary misses use our deterministic
//! Rust rules; no PATH lookup, subprocess, or runtime download occurs.
//! See NOTICE for pinned sources and local changes.
#![allow(dead_code)]
mod lexicon;
mod number;
mod stress;
mod tagger;
mod token;
mod tokenizer;

use lexicon::Lexicon;
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use stress::{apply_stress, CONSONANTS, NON_QUOTE_PUNCTS, PRIMARY_STRESS, SUBTOKEN_JUNKS, VOWELS};
use token::{merge_tokens, MToken, TokenContext};
use tokenizer::TokenOrGroup;

const FALLBACK_CACHE_CAPACITY: usize = 2048;
const MAX_CACHED_WORD_BYTES: usize = 256;

#[derive(Default)]
struct FallbackCache {
    words: HashMap<String, String>,
    order: VecDeque<String>,
}

pub(super) struct G2P {
    lexicon: Lexicon,
    british: bool,
    unk: String,
    overrides: HashMap<String, String>,
    fallback: Mutex<FallbackCache>,
}

pub(super) fn engine(british: bool) -> &'static G2P {
    static US: OnceLock<G2P> = OnceLock::new();
    static GB: OnceLock<G2P> = OnceLock::new();
    (if british { &GB } else { &US }).get_or_init(|| G2P::new(british))
}

pub(super) fn number_words(value: u64) -> String {
    // Used only for parsed clock fields, which are bounded by 59.
    number::int_to_words(value as i64)
}

impl G2P {
    fn new(british: bool) -> Self {
        Self {
            lexicon: Lexicon::with_dialect(british),
            british,
            unk: String::new(),
            overrides: [("walkthrough", "wˈɔkθɹu"), ("walkthroughs", "wˈɔkθɹuz")]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            fallback: Mutex::new(FallbackCache::default()),
        }
    }

    fn fallback_word(&self, word: &str) -> Option<(String, u8)> {
        let key = word.to_string();
        if let Ok(cache) = self.fallback.lock() {
            if let Some(phones) = cache.words.get(&key) {
                return Some((phones.clone(), 1));
            }
        }
        // This word API must not add sentence punctuation. Normalize the legacy
        // rule engine's IPA to the exact symbols in Kokoro's training vocabulary.
        let phones = super::english_g2p::phonemize_word_for_model(word, self.british);
        if key.len() <= MAX_CACHED_WORD_BYTES {
            if let Ok(mut cache) = self.fallback.lock() {
                if !cache.words.contains_key(&key) {
                    if cache.words.len() == FALLBACK_CACHE_CAPACITY {
                        if let Some(oldest) = cache.order.pop_front() {
                            cache.words.remove(&oldest);
                        }
                    }
                    cache.order.push_back(key.clone());
                    cache.words.insert(key, phones.clone());
                }
            }
        }
        Some((phones, 1))
    }

    /// Full pipeline: text -> phoneme string.
    ///
    /// Mirrors misaki `G2P.__call__()` from en.py:679-738.
    pub fn convert(&self, text: &str) -> String {
        // 1. Tokenize and POS-tag (embedded perceptron tagger)
        let tokens = tokenizer::tokenize(text);

        // 2. fold_left: merge non-head tokens
        let tokens = tokenizer::fold_left(tokens);

        // 3. retokenize: subtokenize, handle punctuation/currency
        let mut items = tokenizer::retokenize(tokens);

        // 4. Right-to-left resolution with TokenContext
        let mut ctx = TokenContext::default();

        for item in items.iter_mut().rev() {
            match item {
                TokenOrGroup::Single(ref mut w) => {
                    self.resolve_single_token(w, &ctx);
                    ctx = Self::token_context(&ctx, w.phonemes.as_deref(), w);
                }
                TokenOrGroup::Group(ref mut group) => {
                    self.resolve_group(group, &ctx);
                    if let Some(first) = group.first() {
                        ctx = Self::token_context(&ctx, first.phonemes.as_deref(), first);
                    }
                }
            }
        }

        // 5. Merge groups into single tokens
        let tokens: Vec<MToken> = items
            .into_iter()
            .map(|item| match item {
                TokenOrGroup::Single(tok) => tok,
                TokenOrGroup::Group(group) => merge_tokens(&group, Some(&self.unk)),
            })
            .collect();

        // 6. Legacy conversion: ɾ->T, ʔ->t
        let result: String = tokens
            .iter()
            .map(|tk| {
                let ps = match &tk.phonemes {
                    Some(p) => p.replace('ɾ', "T").replace('ʔ', "t"),
                    None => self.unk.clone(),
                };
                format!("{}{}", ps, tk.whitespace)
            })
            .collect();

        result
    }

    /// Resolve a single (non-grouped) token.
    fn resolve_single_token(&self, w: &mut MToken, ctx: &TokenContext) {
        if w.phonemes.is_some() {
            return;
        }

        // Check custom overrides before the lexicon and Rust rule fallback.
        let lookup_key = w.text.to_lowercase();
        if let Some(ps) = self.overrides.get(&lookup_key) {
            w.phonemes = Some(ps.clone());
            w.underscore.rating = Some(5); // highest priority
            return;
        }
        let (ps, rating) = self.lexicon.call(
            &w.text,
            w.underscore.alias.as_deref(),
            &w.tag,
            w.underscore.stress,
            w.underscore.currency,
            w.underscore.is_head,
            &w.underscore.num_flags,
            ctx,
        );
        if let Some(ps) = ps {
            w.phonemes = Some(ps);
            w.underscore.rating = rating;
            return;
        }

        if let Some((ps, rating)) = self.fallback_word(&w.text) {
            w.phonemes = Some(ps);
            w.underscore.rating = Some(rating);
        }
    }

    /// Resolve a group of subtokens using the left-expand/right-shrink algorithm.
    ///
    /// Ported from en.py:694-731.
    fn resolve_group(&self, group: &mut [MToken], ctx: &TokenContext) {
        let n = group.len();
        let mut left = 0;
        let mut right = n;
        let mut should_fallback = false;

        while left < right {
            let has_existing = group[left..right]
                .iter()
                .any(|tk| tk.underscore.alias.is_some() || tk.phonemes.is_some());

            let (ps, rating) = if has_existing {
                (None, None)
            } else {
                let merged = merge_tokens(&group[left..right], None);
                self.lexicon.call(
                    &merged.text,
                    merged.underscore.alias.as_deref(),
                    &merged.tag,
                    merged.underscore.stress,
                    merged.underscore.currency,
                    merged.underscore.is_head,
                    &merged.underscore.num_flags,
                    ctx,
                )
            };

            if let Some(ps) = ps {
                group[left].phonemes = Some(ps);
                group[left].underscore.rating = rating;
                for x in &mut group[left + 1..right] {
                    x.phonemes = Some(String::new());
                    x.underscore.rating = rating;
                }
                right = left;
                left = 0;
            } else if left + 1 < right {
                left += 1;
            } else {
                right -= 1;
                let tk = &mut group[right];
                if tk.phonemes.is_none() {
                    if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                        tk.phonemes = Some(String::new());
                        tk.underscore.rating = Some(3);
                    } else {
                        should_fallback = true;
                        break;
                    }
                }
                left = 0;
            }
        }

        if should_fallback {
            let merged = merge_tokens(group, None);
            if let Some((ps, rating)) = self.fallback_word(&merged.text) {
                group[0].phonemes = Some(ps);
                group[0].underscore.rating = Some(rating);
                for j in 1..group.len() {
                    group[j].phonemes = Some(String::new());
                    group[j].underscore.rating = group[0].underscore.rating;
                }
            }
        } else {
            Self::resolve_tokens(group);
        }
    }

    /// Update TokenContext based on resolved phonemes and token.
    ///
    /// Ported from en.py:646-650.
    fn token_context(ctx: &TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
        let mut vowel = ctx.future_vowel;

        if let Some(ps) = ps {
            for c in ps.chars() {
                let is_vowel = VOWELS.contains(c);
                let is_consonant = CONSONANTS.contains(c);
                let is_punct = NON_QUOTE_PUNCTS.contains(c);

                if is_vowel || is_consonant || is_punct {
                    vowel = if is_punct { None } else { Some(is_vowel) };
                    break;
                }
            }
        }

        let future_to = matches!(token.text.as_str(), "to" | "To")
            || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));

        TokenContext {
            future_vowel: vowel,
            future_to,
        }
    }

    /// Normalize stress across a group of resolved subtokens.
    ///
    /// Ported from en.py:652-677.
    fn resolve_tokens(tokens: &mut [MToken]) {
        if tokens.is_empty() {
            return;
        }

        let text: String = tokens
            .iter()
            .enumerate()
            .map(|(i, tk)| {
                if i < tokens.len() - 1 {
                    format!("{}{}", tk.text, tk.whitespace)
                } else {
                    tk.text.clone()
                }
            })
            .collect();

        let has_space = text.contains(' ') || text.contains('/');
        let char_classes: std::collections::HashSet<u8> = text
            .chars()
            .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
            .map(|c| {
                if c.is_alphabetic() {
                    0
                } else if c.is_ascii_digit() {
                    1
                } else {
                    2
                }
            })
            .collect();
        let prespace = has_space || char_classes.len() > 1;

        let n = tokens.len();
        for (i, tk) in tokens.iter_mut().enumerate() {
            if tk.phonemes.is_none() {
                let last = i == n - 1;
                if last
                    && tk.text.len() == 1
                    && NON_QUOTE_PUNCTS.contains(tk.text.chars().next().unwrap_or(' '))
                {
                    tk.phonemes = Some(tk.text.clone());
                    tk.underscore.rating = Some(3);
                } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                    tk.phonemes = Some(String::new());
                    tk.underscore.rating = Some(3);
                }
            } else if i > 0 {
                tk.underscore.prespace = prespace;
            }
        }

        if prespace {
            return;
        }

        let indices: Vec<(bool, usize, usize)> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, tk)| {
                tk.phonemes.as_ref().filter(|p| !p.is_empty()).map(|p| {
                    let has_primary = p.contains(PRIMARY_STRESS);
                    let weight = token::stress_weight(Some(p));
                    (has_primary, weight, i)
                })
            })
            .collect();

        if indices.len() == 2 && tokens[indices[0].2].text.len() == 1 {
            let i = indices[1].2;
            if let Some(ref ps) = tokens[i].phonemes {
                tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
            }
            return;
        }

        if indices.len() < 2 {
            return;
        }
        let primary_count: usize = indices.iter().filter(|(b, _, _)| *b).count();
        if primary_count <= indices.len().div_ceil(2) {
            return;
        }

        let mut sorted = indices.clone();
        sorted.sort();
        let half = sorted.len() / 2;
        for &(_, _, i) in &sorted[..half] {
            if let Some(ref ps) = tokens[i].phonemes {
                tokens[i].phonemes = Some(apply_stress(ps, Some(-0.5)));
            }
        }
    }
}
