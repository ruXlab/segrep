use std::error::Error;
use std::fmt;
use tokenizers::tokenizer::{Tokenizer as HfTokenizer};

#[derive(Debug)]
pub enum TokenizerError {
    InitializationError(String),
    TokenizationError(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TokenizerError::InitializationError(msg) => write!(f, "Failed to initialize tokenizer: {}", msg),
            TokenizerError::TokenizationError(msg) => write!(f, "Failed to tokenize text: {}", msg),
        }
    }
}

impl Error for TokenizerError {}

type Result<T> = std::result::Result<T, TokenizerError>;

/// Tokenizes the input string using a pretrained GPT-2 tokenizer (which is BPE-based).
///
/// # Arguments
///
/// * `input` - The string slice to tokenize.
///
/// # Returns
///
/// A vector of token strings.
///
/// # Example
///
/// ```
/// let tokens = segrep::tokenize("Hello, world!").unwrap();
/// println!("{:?}", tokens);
/// // Example output: ["Hello", ",", "Ġworld", "!"]
/// ```
pub fn tokenize(input: &str) -> Result<Vec<String>> {
    // Load the GPT-2 tokenizer from pretrained resources.
    // Note: The first time this is called, the tokenizer files will be downloaded and cached
    let tokenizer = HfTokenizer::from_pretrained("gpt2", None)
        .map_err(|e| TokenizerError::InitializationError(format!(
            "Could not load GPT-2 tokenizer. Please check your internet connection and try again. Error: {}", e
        )))?;

    // Encode the input string. The second argument (true) indicates that the tokenizer should add special tokens if needed.
    let encoding = tokenizer.encode(input, true)
        .map_err(|e| TokenizerError::TokenizationError(format!(
            "Failed to tokenize the input text. Error: {}", e
        )))?;

    Ok(encoding.get_tokens().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_non_empty() {
        let input = "Hello, world!";
        let tokens = tokenize(input).expect("Tokenization should succeed");
        println!("Tokens: {:?}", tokens);
        // Check that we receive at least one token.
        assert!(!tokens.is_empty(), "Token vector should not be empty");
    }

    #[test]
    fn test_tokenize_expected_structure() {
        // For reproducibility note that pretrained tokenizers may have updates,
        // so here we check for minimal expectations.
        let input = "Hello, world!";
        let tokens = tokenize(input).expect("Tokenization should succeed");

        // A typical GPT-2 tokenization of "Hello, world!" is similar to:
        // ["Hello", ",", "Ġworld", "!"]
        // The token "Ġ" is used to indicate a whitespace in GPT-2's BPE.
        // We assert that there are at least 3 tokens and that at least one token starts with the special whitespace marker "Ġ".
        assert!(tokens.len() >= 3, "There should be at least 3 tokens");
        let has_whitespace_token = tokens.iter().any(|t| t.starts_with("Ġ"));
        assert!(has_whitespace_token, "At least one token should indicate a preceding whitespace");
    }

    #[test]
    fn test_error_handling() {
        // Test with an empty string to ensure we handle edge cases
        let result = tokenize("");
        assert!(result.is_ok(), "Empty string should be handled gracefully");
    }
}
