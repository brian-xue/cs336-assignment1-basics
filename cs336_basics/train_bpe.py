import os
import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter

GPT2_PRETOKENIZATION_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_REGEX_PATTERN = None


def get_gpt2_regex_pattern() -> re.Pattern:
    """
    Returns a compiled regex pattern that matches the GPT-2 tokenization scheme.
    """
    global GPT2_REGEX_PATTERN
    if GPT2_REGEX_PATTERN is None:
        GPT2_REGEX_PATTERN = re.compile(GPT2_PRETOKENIZATION_REGEX)
    return GPT2_REGEX_PATTERN


def count_word_freq(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    Counts the frequency of each pre-token sequence in the given text.
    """
    if not text:
        return {}
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    special_token_pattern = re.compile("|".join(re.escape(token.decode("utf-8")) for token in special_token_bytes))
    chunks = special_token_pattern.split(text)
    word_freqs = {}
    for chunk in chunks:
        if not chunk:
            continue
        for match in get_gpt2_regex_pattern().finditer(chunk):
            pretoken = match.group(0)
            if not pretoken:
                continue
            pretoken_bytes = pretoken.encode("utf-8")
            key = tuple(bytes([b]) for b in pretoken_bytes)
            word_freqs[key] = word_freqs.get(key, 0) + 1
    return word_freqs


def pretokenize_text_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
    ) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenizes a chunk of text from the input file and counts the frequency of each pretoken sequence.
    """
    with open(input_path, "rb", encoding="utf-8") as f:
        f.seek(start)
        chunk = f.read(end - start)
    text = chunk.decode("utf-8", errors="ignore")
    return count_word_freq(text, special_tokens)

def serial_pretokenize_text(
    input_path: str,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenizes the entire text file serially and counts the frequency of each pretoken sequence.
    """
    with open(input_path, "r") as f:
        text = f.read()
    return count_word_freq(text, special_tokens)


def parallel_pretokenize_text(
    input_path: str,
    special_tokens: list[str],
    num_processes: int = 4,
) -> dict[tuple[bytes, ...], int]:
    """
    Pretokenizes the entire text file in parallel and counts the frequency of each pretoken sequence.
    """
    if num_processes <= 1:
        return serial_pretokenize_text(input_path, special_tokens)
    special_token = special_tokens[0].encode("utf-8")  # Use the first special token for chunking
    # Find chunk boundaries for parallel processing
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_token)
    
    tasks = [(str(input_path), start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    merged_word_freqs = Counter()
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(pretokenize_text_chunk, tasks)
        for word_freqs in results:
            merged_word_freqs.update(word_freqs)
    return dict(merged_word_freqs)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired vocabulary size.
        special_tokens (list[str]): List of special tokens to include in the vocabulary.
        num_processes (int): Number of processes to use for parallel processing.
    Returns:
        vocab (dict[int, bytes]): The tokenizer vocabulary
        merges (list[tuple[bytes, bytes]]): The list of merges learned during training.
    """
    if vocab_size <= 256 + len(special_tokens):
        raise ValueError(
            f"vocab_size must be greater than 256 + len(special_tokens) ({len(special_tokens)}), got {vocab_size}"
        )
    
    # Initialize the vocabulary with the special tokens and the 256 byte values
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    # pre-tokenization: read the input file and split it into pre-tokens using the GPT-2 regex pattern
    if num_processes is None:
        num_processes = min(8, os.cpu_count() or 1)
    
    file_size = os.path.getsize(input_path)
    if file_size < 1024 * 1024:  # If the file is smaller than 1MB, use serial processing
        word_freqs = serial_pretokenize_text(input_path, special_tokens)
    else:
        word_freqs = parallel_pretokenize_text(input_path, special_tokens, num_processes=num_processes)
    
    if not word_freqs:
        return vocab, []  # Return the initial vocabulary and an empty list of merges if no pre-tokens were found
    

    # --- BPE training loop ---
    merges = [] # list[tuple[bytes, bytes]], The list of merges learned during training
    while next_id < vocab_size:
        # Count pairs of consecutive pre-tokens
        pair_freqs = {} # dict[tuple[bytes, bytes], int], Frequency of each pair of consecutive pre-tokens
        for word, freq in word_freqs.items():
            if len(word) < 2:
                continue
            pairs = zip(word, word[1:])
            for pair in pairs:
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        
        if not pair_freqs:
            break  # No more pairs to merge
        
        # choose the most frequent pair to merge, if ties, break ties by lexicographically largest pair
        (a, b), best_freq = max(pair_freqs.items(), key=lambda x: (x[1], x[0]))
        if best_freq<= 0:
            break  # No more pairs to merge
        
        new_token = a + b
        merges.append((a, b))
        vocab[next_id] = new_token
        next_id += 1

        # Update word_freqs with the new merged token
        new_word_freqs = {} # dict[tuple[bytes, ...], int], Frequency of each pre-token sequence after merging
        for word, freq in word_freqs.items():
            if len(word) < 2:
                new_word_freqs[word] = new_word_freqs.get(word, 0) + freq
                continue
            new_word = [] # list[bytes], The new pre-token sequence after merging
            i = 0
            length = len(word)
            while i < length:
                if i < length - 1 and word[i] == a and word[i + 1] == b:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = new_word_freqs.get(tuple(new_word), 0) + freq
        word_freqs = new_word_freqs
    
    return vocab, merges