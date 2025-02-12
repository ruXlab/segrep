use anyhow::Result;
use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;

/// CrossEncoderSlidingWindowMatcher encapsulates a sequence classification model used as a cross-encoder.
/// It uses a sliding-window approach over a log line to check if any candidate fragment semantically
/// entails a given filter phrase.
pub struct CrossEncoderSlidingWindowMatcher {
    model: SequenceClassificationModel,
    /// Threshold on the entailment score to consider a match.
    threshold: f32,
}

impl CrossEncoderSlidingWindowMatcher {
    /// Creates a new CrossEncoderSlidingWindowMatcher with the given threshold.
    /// Loads the underlying model (typically roberta-large-mnli).
    pub fn new(threshold: f32) -> Result<Self> {
        let model = SequenceClassificationModel::new(Default::default())?;
        Ok(Self { model, threshold })
    }

    /// Compares a filter phrase with a candidate text fragment using the cross-encoder.
    /// Returns a tuple (is_match, score) where is_match is true if the predicted label is
    /// "entailment" and the confidence score is at least the threshold.
    pub fn match_text(&self, filter: &str, candidate: &str) -> Result<(bool, f32)> {
        // Prepare input as (filter, candidate) with owned Strings.
        let inputs = vec!["user is asking about ".to_string() + filter, candidate.to_string()];
        let outputs = self.model.predict(&vec![inputs[0].as_str(), inputs[1].as_str()]);
        let output = &outputs[0];
        // output.text holds the predicted label (e.g., "entailment").
        let predicted_label = output.text.to_lowercase();
        let score = output.score as f32;
        let is_match = predicted_label == "entailment" || score >= self.threshold;
        Ok((is_match, score))
    }
}

/// Checks whether the target text (a log line) contains any contiguous fragment (using a sliding window)
/// that semantically matches the filter phrase. Returns (true, best_score) if at least one candidate
/// fragment is considered a match; otherwise, (false, best_score).
pub fn contains_semantic_fragment(
    matcher: &CrossEncoderSlidingWindowMatcher,
    filter_text: &str,
    text_b: &str,
    window_size: usize,
) -> Result<(bool, f32)> {
    // Split the target log text into words.
    let words: Vec<&str> = text_b.split_whitespace().collect();
    let mut best_score = 0.0;

    // If the text is shorter than the window size, treat the entire text as one candidate.
    if words.len() < window_size {
        let candidate = text_b.to_string();
        let (is_match, score) = matcher.match_text(filter_text, &candidate)?;
        return Ok((is_match, score));
    }

    // Slide a window over the words.
    for window in words.windows(window_size) {
        let candidate = window.join(" ");
        let (is_match, score) = matcher.match_text(filter_text, &candidate)?;
        if score > best_score {
            best_score = score;
        }
        if is_match {
            return Ok((true, best_score));
        }
    }
    Ok((false, best_score))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("authentication failure", "Failed password for invalid user root from 192.168.1.100 port 22 ssh2", true)]
    #[case("authentication failure", "Accepted password for user john from 192.168.1.101 port 22 ssh2", false)]
    #[case("disk error", "I/O error on device sda during read operation", true)]
    #[case("disk error", "Disk usage at 45% on /dev/sda1", false)]
    #[case("connection timeout", "Connection timed out after 5000 ms while reaching api.example.com", true)]
    #[case("connection timeout", "Connection established successfully with remote host", false)]
    #[case("database query failed", "ERROR: relation 'users' does not exist in query execution", true)]
    #[case("database query failed", "Database backup completed successfully", false)]
    #[case("security breach", "ALERT: Intrusion detected on port 22; multiple failed login attempts", true)]
    #[case("security breach", "User logged in without incident", false)]
    #[case("critical failure", "CRITICAL FAILURE: Kernel panic - not syncing: Fatal exception", true)]
    #[case("critical failure", "Routine system check completed without errors", false)]
    #[case("service restart", "Service apache2 restarted successfully after configuration change", true)]
    #[case("service restart", "Service apache2 is running normally", false)]
    #[case("authentication failure", "Multiple failed login attempts detected; invalid user root", true)]
    #[case("disk error", "Disk read operation completed, no errors reported", false)]
    #[case("network congestion", "TCP retransmission detected: possible congestion on eth0", true)]
    #[case("network congestion", "Network interface eth0 is up and running", false)]
    #[case("memory leak", "Memory leak detected: process 1234 consumed 2GB in 1 hour", true)]
    #[case("memory leak", "Memory usage is stable at 512MB", false)]
    #[case("API call", "GET /api/users returned 200 OK", true)]
    #[case("API call", "Deprecated API endpoint /v1/orders no longer available", false)]
    fn test_diverse_log_cases(
        #[case] filter: &str,
        #[case] log_text: &str,
        #[case] expected: bool,
    ) -> Result<()> {
        let matcher = CrossEncoderSlidingWindowMatcher::new(1.0)?;
        let window_size = 1;

        let (is_match, score) = contains_semantic_fragment(&matcher, filter, log_text, window_size)?;

        println!(
            "Filter: '{}'\nLog: '{}'\nWindow Size: {}\nScore: {:.3}\nExpected: {}\nGot: {}\n",
            filter, log_text, window_size, score, expected, is_match
        );

        assert_eq!(
            is_match, expected,
            "Mismatch for filter '{}' and log '{}'",
            filter, log_text
        );

        Ok(())
    }
}
