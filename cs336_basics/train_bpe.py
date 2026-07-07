import os
import regex as re

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


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input file.

    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired vocabulary size.
        special_tokens (list[str]): List of special tokens to include in the vocabulary.

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
    regex_pattern = get_gpt2_regex_pattern()
    word_freqs = {} # dict[tuple[bytes, ...], int], Frequency of each pre-token sequence
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        if not text:
            return vocab, []  # Return empty merges if the input file is empty
    
    # split the text by special tokens first, then apply the regex pattern to each chunk
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    special_token_pattern = re.compile("|".join(re.escape(token.decode("utf-8")) for token in special_token_bytes))
    chunks = special_token_pattern.split(text)

    for chunk in chunks:
        if not chunk:
            continue
        for match in regex_pattern.finditer(chunk):
            pretoken = match.group(0)
            if not pretoken:
                continue
            pretoken_bytes = pretoken.encode("utf-8")
            key = tuple(bytes([b]) for b in pretoken_bytes)
            word_freqs[key] = word_freqs.get(key, 0) + 1
    

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