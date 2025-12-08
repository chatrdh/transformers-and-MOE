import os
import regex as re
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def process_chunk(args):
    """Process a single chunk: read, decode, pre-tokenize"""
    filepath, start, end, special_tokens = args
    
    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # Split on special tokens to prevent merging across boundaries
    if special_tokens:
        # Escape special tokens and join with "|" for regex OR pattern
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
        text_chunks = re.split(special_pattern, chunk)
    else:
        text_chunks = [chunk]
    
    # Pre-tokenize each text chunk separately
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    tokens = []
    
    for text_chunk in text_chunks:
        if text_chunk:  # Skip empty chunks
            pretokens = [match.group() for match in re.finditer(PAT, text_chunk)]
            for token in pretokens:
                tokens.append(list(token.encode('utf-8')))
    
    return tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    merges = []
    vocab = {idx: bytes([idx]) for idx in range(256)}

    # Determine chunk boundaries based on first special token
    num_processes = cpu_count()
    
    with open(input_path, "rb") as f:
        if special_tokens:
            split_token = special_tokens[0].encode('utf-8')
            boundaries = find_chunk_boundaries(f, num_processes, split_token)
        else:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            chunk_size = file_size // num_processes
            boundaries = [i * chunk_size for i in range(num_processes + 1)]
            boundaries[-1] = file_size
    
    # Create arguments for each chunk
    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        token_lists = pool.map(process_chunk, chunk_args)
    
    # Flatten all tokens
    tokens = [token for token_list in token_lists for token in token_list]

    # Build initial pair frequency index and position tracking
    pair_freq = defaultdict(int)
    # Track where each pair appears: pair -> set of (token_idx, position)
    pair_index = defaultdict(set)
    
    for token_idx, token in enumerate(tokens):
        for pos in range(len(token) - 1):
            pair = (token[pos], token[pos + 1])
            pair_freq[pair] += 1
            pair_index[pair].add((token_idx, pos))

    # BPE Loop
    current_vocab_size = 256
    target_size = vocab_size - len(special_tokens)
    
    while current_vocab_size < target_size:
        if not pair_freq:
            break

        # Best pair - break ties lexicographically
        best_pair = max(pair_freq, key=lambda pair: (pair_freq[pair], vocab[pair[0]], vocab[pair[1]]))

        new_token_id = current_vocab_size
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        # Get all positions where this pair occurs
        positions_to_merge = list(pair_index[best_pair])
        
        # Sort by token_idx and position (process in order to handle overlaps)
        positions_to_merge.sort()
        
        # Track affected tokens for incremental update
        affected_tokens = {}  # token_idx -> old_pairs_set
        
        # Collect old pairs from affected tokens
        for token_idx, _ in positions_to_merge:
            if token_idx not in affected_tokens:
                token = tokens[token_idx]
                old_pairs = set()
                for i in range(len(token) - 1):
                    old_pairs.add(((token[i], token[i + 1]), i))
                affected_tokens[token_idx] = old_pairs
        
        # Apply merges (process each token once, handling multiple merges within it)
        for token_idx in affected_tokens.keys():
            token = tokens[token_idx]
            i = 0
            while i < len(token) - 1:
                if token[i] == best_pair[0] and token[i + 1] == best_pair[1]:
                    token[i:i+2] = [new_token_id]
                    # Don't increment i, check the same position again
                else:
                    i += 1
        
        # Update pair frequencies and index incrementally
        # First, remove old pairs from affected tokens
        for token_idx, old_pairs in affected_tokens.items():
            for pair, pos in old_pairs:
                pair_freq[pair] -= 1
                if pair_freq[pair] == 0:
                    del pair_freq[pair]
                    del pair_index[pair]
                else:
                    pair_index[pair].discard((token_idx, pos))
        
        # Then, add new pairs from affected tokens
        for token_idx in affected_tokens.keys():
            token = tokens[token_idx]
            for pos in range(len(token) - 1):
                pair = (token[pos], token[pos + 1])
                pair_freq[pair] += 1
                pair_index[pair].add((token_idx, pos))
    
        current_vocab_size += 1

    # Add special tokens
    for special_token in special_tokens:
        vocab[current_vocab_size] = special_token.encode('utf-8')
        current_vocab_size += 1

    return vocab, merges