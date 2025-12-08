from basic_transformer.bpe import train_bpe
import json

# Train BPE
vocab, merges = train_bpe(
    input_path='data/sample.txt',
    vocab_size=1000,
    special_tokens=["<|endoftext|>"]
)

# Save vocabulary
vocab_serializable = {}
for token_id, token_bytes in vocab.items():
    # Save as list of integers for JSON compatibility
    vocab_serializable[str(token_id)] = list(token_bytes)

with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)

# Save merges
merges_serializable = []
for token1_bytes, token2_bytes in merges:
    merges_serializable.append([list(token1_bytes), list(token2_bytes)])

with open('merges.json', 'w', encoding='utf-8') as f:
    json.dump(merges_serializable, f, ensure_ascii=False, indent=2)

print(f"Vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")
print("Saved vocab.json and merges.json")