use anyhow::Result;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsModel, SentenceEmbeddingsModelType, SentenceEmbeddingsConfig,
};

/// Loads the "all-MiniLM-L6-v2" sentence transformer model and returns a
/// `SentenceEmbeddingsModel` instance. This function will download the model
/// if it isn't cached locally.
pub fn load_model() -> Result<SentenceEmbeddingsModel> {
    let config = SentenceEmbeddingsConfig::from(SentenceEmbeddingsModelType::AllMiniLmL6V2);
    Ok(SentenceEmbeddingsModel::new(config)?)
}

/// Encodes a slice of sentences into their corresponding embedding vectors.
/// 
/// # Arguments
/// 
/// * `sentences` - A slice of string slices that represent the sentences to encode.
/// 
/// # Returns
/// 
/// A vector where each element is an embedding (a vector of f32 values).
/// For the "all-MiniLM-L6-v2" model, each embedding has 384 dimensions.
pub fn encode_sentences(sentences: &[&str]) -> Result<Vec<Vec<f32>>> {
    let model = load_model()?;
    let embeddings = model.encode(sentences)?;
    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_model() {
        // given
        let model = load_model().expect("Model should load without errors");
        
        // when
        let result = model.encode(&["Hello"]);

        // then
        assert!(result.is_ok(), "Encoding should succeed");
    }

    #[test]
    fn test_encode_sentences() {
        // given
        let sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Rust is a systems programming language with great performance.",
            "Python was so untyped that it turned into a duck.",
        ];

        // when
        let embeddings = encode_sentences(&sentences)
            .expect("Encoding of sentences should succeed");

        // then
        assert_eq!(embeddings.len(), sentences.len(), "Mismatch in number of embeddings");

        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(
                emb.len(),
                384,
                "Embedding for sentence {} is expected to have 384 dimensions",
                i
            );
        }
    }
}
