//! Host sampling with one device readback per distribution.

use candle_core::Result;

/// Filter and sample host logits. `random` must be in `[0, 1)`.
pub(super) fn sample_logits(
    logits: &mut [f32],
    temperature: f64,
    top_k: usize,
    top_p: f32,
    random: f32,
) -> Result<u32> {
    if logits.is_empty()
        || !temperature.is_finite()
        || !top_p.is_finite()
        || !(0.0 < top_p && top_p <= 1.0)
        || !(0.0..1.0).contains(&random)
        || logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
    {
        candle_core::bail!("invalid sampling distribution or parameters")
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    if !logits[best].is_finite() {
        candle_core::bail!("sampling distribution has no allowed token")
    }
    if temperature <= 0.0 {
        return Ok(best as u32);
    }

    // Select the threshold in linear time, keeping boundary ties just as the
    // reference top-k logits processor does. No full-vocabulary sort is needed.
    let threshold = if top_k > 0 && top_k < logits.len() {
        let mut scores = logits.to_vec();
        let (_, threshold, _) = scores.select_nth_unstable_by(top_k - 1, |a, b| b.total_cmp(a));
        *threshold
    } else {
        f32::NEG_INFINITY
    };
    let maximum = logits[best] as f64;
    let mut candidates: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite() && **value >= threshold)
        .map(|(index, value)| (index, ((*value as f64 - maximum) / temperature).exp()))
        .collect();
    let total: f64 = candidates.iter().map(|(_, weight)| weight).sum();
    if top_p < 1.0 {
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let target = total * top_p as f64;
        let mut cumulative = 0.0;
        let keep = candidates
            .iter()
            .position(|(_, weight)| {
                cumulative += weight;
                cumulative >= target
            })
            .map_or(candidates.len(), |i| i + 1);
        candidates.truncate(keep);
    }
    let total: f64 = candidates.iter().map(|(_, weight)| weight).sum();
    let target = random as f64 * total;
    let mut cumulative = 0.0;
    for &(token, weight) in &candidates {
        cumulative += weight;
        if cumulative > target {
            return Ok(token as u32);
        }
    }
    Ok(candidates.last().unwrap().0 as u32)
}

pub(super) fn apply_repetition_penalty(logits: &mut [f32], seen: &[bool], penalty: f32) {
    for (logit, &seen) in logits.iter_mut().zip(seen) {
        if seen {
            *logit = if *logit > 0.0 {
                *logit / penalty
            } else {
                *logit * penalty
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_history_penalizes_each_token_once() {
        let mut seen = vec![false; 3];
        for token in [0, 1, 0, 1, 0, 0] {
            seen[token] = true;
        }
        let mut logits = [10.5, -10.0, 7.0];
        apply_repetition_penalty(&mut logits, &seen, 1.05);
        assert!((logits[0] - 10.0).abs() < 1e-5);
        assert!((logits[1] + 10.5).abs() < 1e-5);
        assert_eq!(logits[2], 7.0);
    }

    #[test]
    fn sampling_filters_and_extreme_logits() -> Result<()> {
        assert_eq!(sample_logits(&mut [1., 4., 2.], 0., 0, 1., 0.)?, 1);
        assert_eq!(
            sample_logits(&mut [1000., 999., -1000.], 1., 1, 1., 0.99)?,
            0
        );
        assert_eq!(sample_logits(&mut [1., 4., 2.], 1., 0, 0.01, 0.99)?, 1);
        assert_eq!(
            sample_logits(&mut [f32::NEG_INFINITY, 0.], 1., 0, 1., 0.)?,
            1
        );
        // Both tied top-k entries remain eligible.
        assert_eq!(sample_logits(&mut [3., 3., 0.], 1., 1, 1., 0.75)?, 1);
        assert!(sample_logits(&mut [f32::NEG_INFINITY], 1., 1, 1., 0.).is_err());
        assert!(sample_logits(&mut [f32::NAN], 1., 1, 1., 0.).is_err());
        Ok(())
    }
}
