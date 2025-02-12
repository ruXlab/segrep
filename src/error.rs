use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SegrepError {
    #[error("Failed to initialize tokenizer: {0}")]
    TokenizerInitError(String),

    #[error("Failed to load model from {path}: {message}")]
    ModelLoadError {
        path: PathBuf,
        message: String,
    },

    #[error("Failed to process input: {0}")]
    InputProcessError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("Embedding computation error: {0}")]
    EmbeddingError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Model file not found: {0}")]
    ModelFileNotFound(PathBuf),

    #[error("Invalid model format: {0}")]
    InvalidModelFormat(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),
}

pub type Result<T> = std::result::Result<T, SegrepError>; 