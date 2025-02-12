use anyhow::Result;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsModel, SentenceEmbeddingsModelType,
};
use crate::preprocessing::preprocess;

use crate::model_ops::load_model;

/// A semantic engine that encapsulates the sentence embedding model.
pub struct SemanticEngine {
    model: SentenceEmbeddingsModel,
}

impl SemanticEngine {
    /// Loads model and returns a new SemanticEngine.
    pub fn load() -> Result<Self> {
        let model = load_model()?;
        Ok(SemanticEngine { model })
    }

    /// Computes the semantic similarity between two given texts.
    ///
    /// It first encodes both texts into embeddings and then computes cosine similarity.
    ///
    /// # Returns
    ///
    /// A floating-point value in the range [-1.0, 1.0] representing the cosine similarity.
    /// A value close to 1 indicates very similar semantics.
    pub fn semantic_similarity(&self, text1: &str, text2: &str) -> Result<f32> {
        let sentences = [preprocess(text1), preprocess(text2)];
        let embeddings = self.model.encode(&sentences)?;
        if embeddings.len() != 2 {
            return Err(anyhow::anyhow!(
                "Expected 2 embeddings but got {}",
                embeddings.len()
            ));
        }
        Ok(cosine_similarity(&embeddings[0], &embeddings[1]))
    }

    /// Determines if a text matches a semantic filter based on similarity threshold
    pub fn matches_semantic_filter(&self, filter: &str, text: &str, threshold: f32) -> Result<bool> {
        let similarity = self.semantic_similarity(filter, text)?;
        Ok(similarity >= threshold)
    }
}

/// Computes the cosine similarity between two vectors.
///
/// Cosine similarity is defined as:
///
/// ```ignore
/// similarity = (A · B) / (||A|| * ||B||)
/// ```
///
/// where:
/// - `A` and `B` are the input vectors,
/// - `·` denotes the dot product,
///
/// The result is a value between -1.0 and 1.0,
/// where 1.0 means the vectors point in the same direction.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use rstest::rstest;

    const SIMILARITY_THRESHOLD: f32 = 0.6;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors should yield a similarity of 1.
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v1, &v2) - 1.0).abs() < 1e-6);

        // Orthogonal vectors should yield a similarity of 0.
        let v3 = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&v1, &v3)).abs() < 1e-6);
    }

    #[test]
    fn test_basic_semantic_similarity() {
        let engine = SemanticEngine::load().expect("Failed to load SemanticEngine");

        // Two semantically similar sentences should yield a high similarity.
        let text1 = "The quick brown fox jumps over the lazy dog.";
        let text2 = "A fast, dark-colored fox leaps over a sleepy dog.";
        let sim = engine.semantic_similarity(text1, text2)
            .expect("Failed to compute semantic similarity");
        println!("Similarity (similar texts): {}", sim);
        assert!(sim > 0.7, "Expected similarity > 0.7, got {}", sim);

        // Two dissimilar sentences should yield a lower similarity.
        let text3 = "I love machine learning.";
        let text4 = "The weather is sunny today.";
        let sim2 = engine.semantic_similarity(text3, text4)
            .expect("Failed to compute semantic similarity");
        println!("Similarity (dissimilar texts): {}", sim2);
        assert!(sim2 < 0.5, "Expected similarity < 0.5, got {}", sim2);
    }

    #[rstest]
    #[case("user registration", "user signed up from EMEA", true)]
    #[case("user registration", "404 error - resource /user is not found", false)]
    #[case("x11", "wayland failed to start", true)]
    #[case("password reset", "User requested a password reset via email", true)]
    #[case("password reset", "User updated his profile picture", false)]
    #[case("server error", "Internal server error occurred during API processing", true)]
    #[case("server error", "Connection established with the server successfully", false)]
    #[case("login attempt", "User failed login attempt due to incorrect password", true)]
    #[case("login attempt", "User logged in successfully", false)]
    #[case("data export", "Scheduled export of CSV data initiated by admin", true)]
    #[case("data export", "Data import process encountered malformed JSON input", false)]
    #[case("performance degradation", "System performance degraded under heavy load", true)]
    #[case("performance degradation", "System performance is optimal with no noticeable delays", false)]
    #[case("API call", "Successful API call to /v1/users endpoint returned status code 200", true)]
    #[case("API call", "Deprecated API endpoint /v1/orders no longer available", false)]
    #[case("gpu driver", "Nvidia GPU driver experienced a critical crash during rendering", true)]
    #[case("gpu driver", "GPU driver updated with new bug fixes and improvements", false)]
    fn test_semantic_filters(
        #[case] filter: &str,
        #[case] text: &str,
        #[case] expected_match: bool,
    ) {
        let engine = SemanticEngine::load().expect("Failed to load SemanticEngine");
        let matches = engine
            .matches_semantic_filter(filter, text, SIMILARITY_THRESHOLD)
            .expect("Failed to check semantic filter");
        
        println!("\nFilter: '{}'\nText: '{}'\nExpected Match: {}\nActual Match: {}", 
                 filter, text, expected_match, matches);
        
        if let Ok(similarity) = engine.semantic_similarity(filter, text) {
            println!("Similarity Score: {:.3}", similarity);
        }
        
        assert_eq!(matches, expected_match, 
            "Semantic filter '{}' failed for text '{}'. Expected match: {}, got: {}", 
            filter, text, expected_match, matches);
    }

    #[test]
    fn test_edge_cases() {
        let engine = SemanticEngine::load().expect("Failed to load SemanticEngine");
        
        // Empty strings
        let sim = engine.semantic_similarity("", "")
            .expect("Failed to compute similarity for empty strings");
        assert!(sim.is_finite(), "Similarity for empty strings should be a finite number");
        
        // Very long text
        let long_text1 = "a ".repeat(1000);
        let long_text2 = "b ".repeat(1000);
        let sim = engine.semantic_similarity(&long_text1, &long_text2)
            .expect("Failed to compute similarity for long texts");
        assert!(sim.is_finite(), "Similarity for long texts should be a finite number");
        
        // Special characters
        let special_text1 = "!@#$%^&*()_+";
        let special_text2 = "{}[]|\\:;\"'<>,.?/";
        let sim = engine.semantic_similarity(special_text1, special_text2)
            .expect("Failed to compute similarity for special characters");
        assert!(sim.is_finite(), "Similarity for special characters should be a finite number");
    }
}
