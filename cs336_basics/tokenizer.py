import regex as re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import os
from typing import Iterable, Iterator
import pickle

GPT2_PRETOKENIZATION_REGEX = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
            self, 
            vocab: dict[int, bytes], 
            merges : list[tuple[bytes, bytes]], 
            special_tokens:list[str] |None =None
            ):
        """
        Initializes the Tokenizer with a vocabulary, merges, and optional special tokens.
        """
        self.vocab = vocab
        self.merges = merges

        self.byte_to_id = {v: k for k, v in vocab.items()} # dict[bytes, int], Maps byte sequences to their corresponding token IDs
        self.merge_rank = {pair: i for i, pair in enumerate(merges)} # dict[tuple[bytes, bytes], int], Maps pairs of byte sequences to their merge rank

        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_bytes = [] # list[bytes], Stores the byte representations of special tokens
        self.special_tokens_id = {} # dict[str, int], Maps string representations of special tokens to their corresponding token IDs

        self.rx = re.compile(GPT2_PRETOKENIZATION_REGEX)

        if self.special_tokens:
            # append special tokens to the vocabulary and update byte_to_id mapping
            for token in self.special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.byte_to_id:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
                    self.byte_to_id[token_bytes] = new_id
                self.special_tokens_bytes.append(token_bytes)
                self.special_tokens_id[token] = self.byte_to_id[token_bytes]

            # sort the special tokens by length in descending order to ensure longer tokens are matched first
            sorted_special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
            self._special_re = re.compile("|".join(re.escape(token) for token in sorted_special_tokens))
            self._max_special_token_length = max(len(token) for token in sorted_special_tokens)
        else:
            self._special_re = None
            self._max_special_token_length = 0

        self._bpe_cache: dict[bytes, list[bytes]] = {} # dict[bytes, list[bytes]], Caches the BPE-encoded byte sequences for byte sequences

    @classmethod
    def from_files(
            cls, 
            vocab_path: str, 
            merges_path: str, 
            special_tokens: list[str] |None =None
            ):
        """
        Initializes the Tokenizer from vocabulary and merges files.
        """
        vocab = {}
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        merges = []
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string of text into a list of token IDs.
        """
        if not text:
            return []
        ids: list[int] = []
        # Check for special tokens in the text
        if not self.special_tokens:
            ids.extend(self._encode_plain_text(text))
            return ids
        else:
            last_index = 0
            for matches in self._special_re.finditer(text):
                start, end = matches.span()
                # Encode the text before the special token
                if start > last_index:
                    ids.extend(self._encode_plain_text(text[last_index:start]))
                # Encode the special token
                s = matches.group(0)
                ids.append(self.special_tokens_id[s])
                last_index = end
            # Encode any remaining text after the last special token
            if last_index < len(text):
                ids.extend(self._encode_plain_text(text[last_index:]))
            return ids

   
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        buf = ""
        for chunk in iterable:
            if not chunk:
                continue
            buf += chunk
            while True:
                matches = list(self.rx.finditer(buf))
                if len(matches) <= 1:
                    break

                # keep the last match in the buffer for the next iteration
                cutoff = matches[-1].start()
                if cutoff <=0:
                    break
                process_chunk = buf[:cutoff]
                buf = buf[cutoff:]
                for token_id in self.encode(process_chunk):
                    yield token_id
        
        # flush the remaining buffer
        for token_id in self.encode(buf):
            yield token_id


    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        # Convert token IDs back to their byte representations
        byte_sequences = [self.vocab[i] for i in ids]
        # Concatenate the byte sequences and decode to a UTF-8 string
        return b"".join(byte_sequences).decode("utf-8", errors="replace")

    def _encode_plain_text(self, text: str) -> list[int]:
        """
        Encodes plain text (without special tokens) into a list of token IDs.
        """
        if not text:
            return []
        # Convert the text to bytes
        output_ids: list[int] = []
        for m in self.rx.finditer(text):
            piece = m.group(0)
            if not piece:
                continue
            # Convert the piece to bytes
            piece_bytes = piece.encode("utf-8")
            # Apply BPE merging to the piece and convert to token IDs
            for token in self._bpe_merge(piece_bytes):
                output_ids.append(self.byte_to_id[token])
        return output_ids

    def _bpe_merge(self, tokens: bytes) -> list[bytes]:
        """
        Applies BPE merging to byte tokens and returns the corresponding byte sequences.
        """
        cached_val = self._bpe_cache.get(tokens)
        if cached_val is not None:
            return cached_val
        
        # Start with single-character tokens
        word = [bytes([b]) for b in tokens] # list[bytes]
        if len(word) <= 1:
            self._bpe_cache[tokens] = word
            return word
        
        # Repeatedly merge the most frequent pair of bytes until no more merges can be done
        while True:
            best_pair = None
            best_rank = float("inf")

            prev = word[0]
            for curr in word[1:]:
                pair = (prev, curr)
                rank = self.merge_rank.get(pair)
                if rank is not None and (rank < best_rank):
                    best_pair = pair
                    best_rank = rank
                prev = curr
            
            if best_pair is None:
                break
            # Merge the best pair
            a, b = best_pair
            new_token = a + b

            merged_word: list[bytes] = []
            i = 0
            L = len(word)
            while i < L:
                if i < L - 1 and word[i] == a and word[i + 1] == b:
                    merged_word.append(new_token)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1
            word = merged_word
            if len(word) <= 1:
                break
        self._bpe_cache[tokens] = word
        return word
        
