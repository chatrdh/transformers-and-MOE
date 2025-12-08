from basic_transformer.tokenizer import Tokenizer

# Load your trained tokenizer
tokenizer = Tokenizer.from_files(
    vocab_filepath='vocab.json',
    merges_filepath='merges.json',
    special_tokens=["<|endoftext|>"]
)

# Example 1: Basic encoding/decoding
text = "The quick brown fox jumps over the lazy dog."
print(f"Original text: {text}")

token_ids = tokenizer.encode(text)
print(f"Token IDs: {token_ids}")
print(f"Number of tokens: {len(token_ids)}")

decoded = tokenizer.decode(token_ids)
print(f"Decoded text: {decoded}")
print(f"Match: {text == decoded}")

# Example 2: With special tokens
story = "Once upon a time, there was a little girl.<|endoftext|>The end."
token_ids = tokenizer.encode(story)
print(f"\nWith special token: {token_ids}")
decoded = tokenizer.decode(token_ids)
print(f"Decoded: {decoded}")

# Example 3: Processing a file
def tokenize_file(input_file, output_file):
    """Tokenize a text file and save token IDs"""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    token_ids = tokenizer.encode(text)
    
    # Save as binary or JSON
    import json
    with open(output_file, 'w') as f:
        json.dump(token_ids, f)
    
    print(f"Tokenized {len(text)} chars into {len(token_ids)} tokens")
    return token_ids

# tokenize_file('input.txt', 'tokens.json')

# Example 4: Batch processing with encode_iterable
sentences = [
    "This is the first sentence.",
    "Here's another one.",
    "And a third sentence."
]

all_tokens = list(tokenizer.encode_iterable(sentences))
print(f"\nBatch tokenization: {len(all_tokens)} total tokens")