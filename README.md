# segrep: Semantic Search CLI Tool

segrep is a high-performance CLI utility written in Rust that performs semantic search on text (e.g., log lines) from streamed standard input. By leveraging embedding-based tokenization and semantic similarity computation, segrep efficiently retrieves lines that match natural language queries.

For instance, given the following log line:

```
2024-01-01 12:00:00.000 [INFO] [com.example.app] - User from EMEA region registered
```

Can be retrieved by running the following command:

```
segrep "signup from europe"
```

## Model Setup

Before building the project, you need to download the required model files. You can do this in two ways:

### Option 1: Using the Setup Script

Run the provided setup script:
```bash
./scripts/setup_models.sh
```

This will:
1. Create a `models` directory
2. Download the BERT tokenizer vocabulary
3. Download and extract the MiniLM model weights
4. Verify the downloads

### Option 2: Manual Setup

1. Create a models directory:
   ```bash
   mkdir -p models
   ```

2. Download the BERT tokenizer:
   ```bash
   curl -L https://huggingface.co/bert-base-uncased/raw/main/vocab.txt \
        -o models/bert-base-uncased-vocab.txt
   ```

3. Download the MiniLM model:
   ```bash
   curl -L https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin \
        -o models/all-MiniLM-L6-v2/pytorch_model.bin
   ```

4. Download the model configuration:
   ```bash
   curl -L https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/config.json \
        -o models/all-MiniLM-L6-v2/config.json
   ```

The model files should be organized as follows:
```
models/
├── bert-base-uncased-vocab.txt
└── all-MiniLM-L6-v2/
    ├── config.json
    └── pytorch_model.bin
```

## LibTorch Setup

This project requires LibTorch (PyTorch C++ library). You have several options to set it up:

1. **Use existing PyTorch installation (Recommended)**:
   ```bash
   # If you have PyTorch installed via pip
   export LIBTORCH_USE_PYTORCH=1
   cargo test
   ```

2. **Manual LibTorch installation**:
   ```bash
   # Download and extract LibTorch
   wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcpu.zip
   unzip libtorch-cxx11-abi-shared-with-deps-2.2.1+cpu.zip
   export LIBTORCH=/path/to/libtorch
   cargo test
   ```

3. **System package installation**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libtorch-dev

   # Arch Linux
   sudo pacman -S python-pytorch
   ```

If you don't have PyTorch installed, you can install it with:
```bash
pip install torch
export LIBTORCH_USE_PYTORCH=1
```

## Features

- **Streaming Input:** Reads data from stdin in either streaming or batched mode.
- **Interchangeable Tokenizers:** Supports multiple tokenization methods including Hugging Face, SentencePiece, and a unified adapter via the rust-tokenizers crate.
- **Embedding Generation:** Converts token sequences into fixed-length vectors using lightweight models (via tch-rs or ONNX Runtime).
- **Semantic Search:** Computes cosine similarity between query and data embeddings to filter and return relevant lines.
- **Optimized Performance:** Built with Rust and optimized for release builds with LTO and minimized codegen units.

## Modules Overview

1. **CLI & Input Module**
   - Parses command-line arguments using [clap](https://github.com/clap-rs/clap).
   - Handles efficient read from stdin.

2. **Tokenization Module**
   - Defines a `Tokenizer` trait with methods for tokenization (and optionally detokenization).
   - Implements three alternative tokenizers: Hugging Face, SentencePiece, and a unified adapter.

3. **Embedding Module**
   - Integrates embedding models to convert tokenized text into vector representations.
   - Supports Rust bindings for PyTorch (tch-rs) or ONNX Runtime.

4. **Semantic Search Module**
   - Computes cosine similarity between query embeddings and input line embeddings.
   - Filters results based on a configurable similarity threshold.

5. **Output Module**
   - Formats and prints matching results (with optional similarity scores) to stdout.

## Testing Strategy

- **Unit Tests:** 
  - Validates tokenizers, embedding functions, and similarity computations using Rust's built-in test framework.

- **Integration Tests:** 
  - Uses [assert_cmd](https://github.com/assert-rs/assert_cmd) to simulate CLI behavior with sample inputs.
  - Includes text-based tests that simulate real-world log data.

## Benchmarking

- Benchmarks are implemented with [Criterion](https://github.com/bheisler/criterion.rs).
- Both microbenchmarks (for tokenization, embedding, similarity) and full pipeline benchmarks are provided.
- Execute benchmarks using:
  ```
  cargo bench
  ```

## Installation and Usage

1. Clone the repository:
   ```
   git clone <repository-url>
   cd segrep
   ```

2. Build in release mode:
   ```
   cargo build --release
   ```

3. Run with sample input:
   ```
   cat sample.log | ./target/release/segrep --query "error occurred"
   ```

## Development Roadmap

- **Phase 1:** Setup basic CLI and streaming input.
- **Phase 2:** Implement modular tokenization layers with multiple tokenizer options.
- **Phase 3:** Integrate embedding generation and semantic search.
- **Phase 4:** Develop comprehensive unit, integration, and text-based tests.
- **Phase 5:** Benchmark critical components and optimize performance.
- **Phase 6:** Finalize packaging, documentation, and CI/CD integration.

## Packaging and Deployment

- Release builds use Link-Time Optimization (LTO) and a single codegen unit for improved performance.
- Distribution as a standalone binary with optional cross-compilation for various platforms.

## License

Ensure that all dependencies have permissive licenses (e.g., Apache 2.0) to facilitate broad distribution of the final binary.

## Contributing

Contributions are welcome! Please refer to the CURSOR_RULES.md for coding guidelines, commit standards, and best practices for maintaining the code quality and performance of segrep. 