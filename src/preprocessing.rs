/// Preprocesses the given input string for tokenization by:
/// - Converting all characters to lowercase.
/// - Collapsing multiple consecutive whitespace characters into a single space.
/// - Trimming leading and trailing whitespace.
///
/// This version avoids using regex for performance reasons.
///
/// # Examples
///
/// ```
/// let input = "  Hello,   World!  This is    Rust. ";
/// let normalized = segrep::preprocessing::preprocess(input);
/// assert_eq!(normalized, "hello, world! this is rust.");
/// ```
pub fn preprocess(input: &str) -> String {
    // Convert the entire input to lowercase.
    let lower = input.to_lowercase();
    let mut result = String::with_capacity(lower.len());
    let mut prev_is_space = false;

    // Iterate over each character.
    for ch in lower.chars() {
        if ch.is_whitespace() {
            // Only add a single space if the previous character was not whitespace.
            if !prev_is_space {
                result.push(' ');
                prev_is_space = true;
            }
        } else {
            result.push(ch);
            prev_is_space = false;
        }
    }
    // Trim any leading or trailing whitespace and return.
    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parameterized test cases for the `preprocess` function.
    /// Each tuple is (input, expected_output).
    #[test]
    fn test_preprocess() {
        let test_cases = vec![
            ("Hello, World!", "hello, world!"),
            ("  Multiple   Spaces  ", "multiple spaces"),
            ("Normalize THIS TEXT.", "normalize this text."),
            ("User's registration", "user's registration"),
            ("Keep special characters: @$%^&*()!", "keep special characters: @$%^&*()!"),
            ("MixedCASE with NUMBERS 123!", "mixedcase with numbers 123!"),
            ("   Leading and trailing    ", "leading and trailing"),
            ("Tabs\tand newlines\nshould collapse", "tabs and newlines should collapse"),
        ];

        for (input, expected) in test_cases {
            let output = preprocess(input);
            assert_eq!(output, expected, "Preprocessing failed for input: {:?}", input);
        }
    }
}
