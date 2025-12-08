import regex as re
from typing import Iterator, Iterable
import json


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from vocabulary, merges, and special tokens.
        """
        # Store vocab exactly as provided
        self.vocab = vocab.copy()
        self.merges = merges
        
        # Create reverse vocab for encoding
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Build merge rules
        self.merge_rules = {}
        
        for token1_bytes, token2_bytes in self.merges:
            id1 = self.inverse_vocab.get(token1_bytes)
            id2 = self.inverse_vocab.get(token2_bytes)
            
            if id1 is not None and id2 is not None:
                merged_bytes = token1_bytes + token2_bytes
                merged_id = self.inverse_vocab.get(merged_bytes)
                
                if merged_id is not None:
                    self.merge_rules[(id1, id2)] = merged_id
        
        # Handle special tokens - SORT BY LENGTH (longest first)
        self.special_tokens = sorted(special_tokens if special_tokens else [], key=len, reverse=True)
        self.special_token_ids = {}
        
        if self.vocab:
            next_id = max(self.vocab.keys()) + 1
        else:
            next_id = 0
        
        for special_token in self.special_tokens:
            special_bytes = special_token.encode('utf-8')
            
            if special_bytes in self.inverse_vocab:
                self.special_token_ids[special_token] = self.inverse_vocab[special_bytes]
            else:
                self.vocab[next_id] = special_bytes
                self.inverse_vocab[special_bytes] = next_id
                self.special_token_ids[special_token] = next_id
                next_id += 1
        
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Construct a Tokenizer from serialized vocabulary and merges files.
        """
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        vocab = {}
        for key, value in vocab_data.items():
            token_id = int(key)
            if isinstance(value, list):
                vocab[token_id] = bytes(value)
            elif isinstance(value, str):
                vocab[token_id] = value.encode('latin-1')
            else:
                vocab[token_id] = bytes(value)
        
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_data = json.load(f)
        
        merges = []
        for merge_item in merges_data:
            token1, token2 = merge_item
            
            if isinstance(token1, list):
                token1_bytes = bytes(token1)
            elif isinstance(token1, str):
                token1_bytes = token1.encode('latin-1')
            else:
                token1_bytes = bytes(token1)
            
            if isinstance(token2, list):
                token2_bytes = bytes(token2)
            elif isinstance(token2, str):
                token2_bytes = token2.encode('latin-1')
            else:
                token2_bytes = bytes(token2)
            
            merges.append((token1_bytes, token2_bytes))
        
        return cls(vocab, merges, special_tokens)
    
    def _apply_merges(self, token_ids: list[int]) -> list[int]:
        """
        Apply BPE merges to a list of token IDs.
        """
        while len(token_ids) >= 2:
            # Find all pairs and their positions
            pairs = [(i, (token_ids[i], token_ids[i + 1])) for i in range(len(token_ids) - 1)]
            
            # Filter to only pairs that have merge rules
            valid_pairs = [(i, pair) for i, pair in pairs if pair in self.merge_rules]
            
            if not valid_pairs:
                break
            
            # Find the pair with the lowest merged_id (highest priority)
            best_idx, best_pair = min(valid_pairs, key=lambda x: self.merge_rules[x[1]])
            
            # Apply the merge at best_idx
            merged_id = self.merge_rules[best_pair]
            token_ids = token_ids[:best_idx] + [merged_id] + token_ids[best_idx + 2:]
        
        return token_ids
    
    def encode(self, text: str) -> list[int]:
        """
        Encode text into a sequence of token IDs.
        """
        # Handle special tokens by splitting on them (longest first)
        if self.special_tokens:
            # Create pattern with special tokens sorted by length (already done in __init__)
            # This ensures longer tokens are matched first
            special_pattern = '(' + '|'.join(re.escape(token) for token in self.special_tokens) + ')'
            parts = re.split(special_pattern, text)
        else:
            parts = [text]
        
        all_ids = []
        
        for part in parts:
            if not part:
                continue
            
            # Check if this part is a special token
            if part in self.special_token_ids:
                all_ids.append(self.special_token_ids[part])
                continue
            
            # Pre-tokenize using regex
            pretokens = re.findall(self.PAT, part)
            
            for pretoken in pretokens:
                # Convert to UTF-8 bytes
                pretoken_bytes = pretoken.encode('utf-8')
                
                # Convert bytes to token IDs by looking up each byte in inverse_vocab
                token_ids = []
                for byte_val in pretoken_bytes:
                    # Look up the single byte in inverse_vocab
                    single_byte = bytes([byte_val])
                    if single_byte in self.inverse_vocab:
                        token_ids.append(self.inverse_vocab[single_byte])
                    else:
                        # Fallback: use byte value as ID
                        token_ids.append(byte_val)
                
                # Apply BPE merges
                merged_ids = self._apply_merges(token_ids)
                all_ids.extend(merged_ids)
        
        return all_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings, yielding token IDs one at a time.
        """
        for text in iterable:
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_sequences = []
        
        for token_id in ids:
            if token_id in self.vocab:
                byte_sequences.append(self.vocab[token_id])
        
        all_bytes = b''.join(byte_sequences)
        
        try:
            text = all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = all_bytes.decode('utf-8', errors='replace')
        
        return text