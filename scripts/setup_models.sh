#!/bin/bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to download with progress
download_with_progress() {
    local url=$1
    local output=$2
    local description=$3
    
    echo -e "${YELLOW}Downloading ${description}...${NC}"
    curl -L --progress-bar "$url" -o "$output"
    
    if [ $? -ne 0 ] || [ ! -f "$output" ]; then
        echo -e "${RED}Failed to download ${description}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Successfully downloaded ${description}${NC}"
}

echo -e "${YELLOW}Setting up segrep models...${NC}"

# Create models directory
echo "Creating models directory..."
mkdir -p models/all-MiniLM-L6-v2

# Download files with progress bars
download_with_progress \
    "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt" \
    "models/bert-base-uncased-vocab.txt" \
    "BERT tokenizer vocabulary"

download_with_progress \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin" \
    "models/all-MiniLM-L6-v2/pytorch_model.bin" \
    "MiniLM model weights"

download_with_progress \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/raw/main/config.json" \
    "models/all-MiniLM-L6-v2/config.json" \
    "model configuration"

# Verify file sizes
echo "Verifying downloads..."
VOCAB_SIZE=$(stat -f%z "models/bert-base-uncased-vocab.txt" 2>/dev/null || stat -c%s "models/bert-base-uncased-vocab.txt")
MODEL_SIZE=$(stat -f%z "models/all-MiniLM-L6-v2/pytorch_model.bin" 2>/dev/null || stat -c%s "models/all-MiniLM-L6-v2/pytorch_model.bin")
CONFIG_SIZE=$(stat -f%z "models/all-MiniLM-L6-v2/config.json" 2>/dev/null || stat -c%s "models/all-MiniLM-L6-v2/config.json")

# Minimum expected sizes (in bytes)
MIN_VOCAB_SIZE=100000   # ~100KB
MIN_MODEL_SIZE=50000000 # ~50MB
MIN_CONFIG_SIZE=100     # ~100B

if [ "$VOCAB_SIZE" -lt "$MIN_VOCAB_SIZE" ] || \
   [ "$MODEL_SIZE" -lt "$MIN_MODEL_SIZE" ] || \
   [ "$CONFIG_SIZE" -lt "$MIN_CONFIG_SIZE" ]; then
    echo -e "${RED}Warning: One or more files appear to be incomplete${NC}"
    echo "Expected minimum sizes:"
    echo "- Vocabulary: 100KB"
    echo "- Model: 50MB"
    echo "- Config: 100B"
    echo "Got:"
    echo "- Vocabulary: $(($VOCAB_SIZE/1024))KB"
    echo "- Model: $(($MODEL_SIZE/1024/1024))MB"
    echo "- Config: ${CONFIG_SIZE}B"
    exit 1
fi

echo -e "${GREEN}Setup completed successfully!${NC}"
echo "Model files are ready to use with segrep" 