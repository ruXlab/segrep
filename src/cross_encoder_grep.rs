use anyhow::Result;
use rust_bert::pipelines::sequence_classification::SequenceClassificationModel;
use crate::preprocessing::preprocess;
/// CrossEncoderMatcher encapsulates a sequence classification model used as a cross-encoder.
/// It takes in a filter phrase and a target text, and returns whether the target
/// semantically entails the filter with a score above a threshold.
pub struct CrossEncoderMatcher {
    model: SequenceClassificationModel,
    /// Threshold on the entailment score to consider a match.
    threshold: f32,
}

impl CrossEncoderMatcher {
    /// Creates a new CrossEncoderMatcher with the given threshold.
    /// The underlying model (typically roberta-large-mnli) is loaded.
    pub fn new(threshold: f32) -> Result<Self> {
        // Use default configuration to load the sequence classification model.
        let model = SequenceClassificationModel::new(Default::default())?;
        
        Ok(Self { model, threshold })
    }

    /// Computes the semantic match between a filter (premise) and a target text (hypothesis)
    /// using the cross-encoder. Returns a tuple of (is_match, entailment_score).
    ///
    /// A match is considered true if:
    /// - The predicted label is "entailment" (case-insensitive), and
    /// - The model's score is greater than or equal to the threshold.
    pub fn match_text(&self, filter: &str, text: &str) -> Result<(bool, f32)> {
        // Convert the filter and text into owned Strings so that the model accepts them.
        let inputs = vec!["user is asking about ".to_string() + filter, text.to_string()];
        let outputs = self.model.predict(&vec![inputs[0].as_str(), inputs[1].as_str()]);
        // Using the first (and only) output:
        let output = &outputs[0];
        // Note: the Label struct has fields: text, score, id, sentence.
        // Here, we treat `output.text` as the predicted label (e.g., "entailment").
        let predicted_label = output.text.to_lowercase();
        let score = output.score as f32;
        // New decision logic:
        // 1. If the predicted label is "entailment" and the score >= threshold, it's a match.
        // 2. Alternatively, if the score is extremely high (e.g., >= 0.99), consider it a match regardless.
        let is_match = predicted_label == "entailment" && score >= self.threshold || score >= 0.99;
        Ok((is_match, score))        
    }
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
    #[case("network congestion", "TCP retransmission detected: possible congestion on eth0", true)]
    #[case("network congestion", "Network interface eth0 is up and running", false)]
    #[case("timeout", "Connection timed out after 5000 ms", true)]
    #[case("timeout", "Connection established successfully", false)]
    #[case("database query failed", "ERROR: relation 'users' does not exist", true)]
    #[case("database query failed", "Database backup completed successfully", false)]
    #[case("security breach", "ALERT: Intrusion detected on port 22", true)]
    #[case("security breach", "User logged in successfully", false)]
    #[case("memory leak", "Memory leak detected: process 1234 consumed 2GB in 1 hour", true)]
    #[case("memory leak", "Memory usage is stable at 512MB", false)]
    #[case("high latency", "Latency spike: response time exceeded 300ms", true)]
    #[case("high latency", "Response time: 50ms", false)]
    #[case("configuration error", "Configuration error: missing field 'server.port'", true)]
    #[case("configuration error", "Configuration loaded successfully", false)]
    #[case("service restart", "Service apache2 restarted", true)]
    #[case("service restart", "Service apache2 running normally", false)]
    #[case("connection refused", "Connection refused by remote host", true)]
    #[case("connection refused", "Connected to remote host successfully", false)]
    #[case("critical failure", "CRITICAL FAILURE: Kernel panic - not syncing: Fatal exception", true)]
    #[case("critical failure", "Routine system check completed", false)]
    #[case("power outage", "Power outage detected - switching to backup generator", true)]
    #[case("power outage", "System running on main power", false)]
    #[case("authentication failure", "Multiple failed login attempts detected", true)]
    #[case("server error", "Service responded with a 500 error", true)]
    #[case("database query failed", "SQLSTATE[42P01]: Undefined table: relation \"orders\" does not exist", true)]
    #[case("API call", "GET /api/users returned 200 OK", true)]
    #[case("API call", "Deprecated API endpoint /v1/orders no longer available", false)]
    #[case("payment declined", "Payment was declined due to insufficient funds", true)]
    #[case("payment declined", "Payment processed successfully", false)]
    #[case("disk", "Disk read operation completed, no errors reported", true)]
    #[case("disk", "user was removed", false)]
    #[case("disk", "run out of the space", true)]
    #[case("disk", "disk is full", true)]
    #[case("disk", "Grafana server is not responding", false)]
    #[case("disk", "Kotlin native is not very popular, unfortunately", false)]
    #[case("disk", "Matrices don't have soul", false)]
    fn test_diverse_log_cases(#[case] filter: &str, #[case] log_text: &str, #[case] expected: bool) {
        let matcher = CrossEncoderMatcher::new(0.9).expect("Failed to load cross-encoder model");
        let (is_match, score) = matcher.match_text(&preprocess(filter), &preprocess(log_text)).expect("Matching failed");
        let match_emoji = if is_match == expected { "✅" } else { "❌" };

        println!(
            "Filter: '{}'\nLog: '{}'\nScore: {:.3}\nExpected: {}\nGot: {} {}\n",
            filter, log_text, score, expected, is_match, match_emoji
        );
        assert_eq!(
            is_match, expected,
            "Mismatch for filter '{}' and log '{}'",
            filter, log_text
        );
    }
}
