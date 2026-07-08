from cs336_basics.train_bpe import train_bpe
import pathlib
import time
import os, psutil, pickle

def train_bpe_on_tinystories():
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "workspace"
    os.makedirs(output_dir, exist_ok=True)

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    proc = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes = 8)
    end_time = time.perf_counter()

    rss_memory = proc.memory_info().rss / (1024 * 1024*1024)  # in GB
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Peak memory usage: {rss_memory:.2f} GB")
    vocab_path = os.path.join(output_dir, "vocab_tinystories.pkl")
    merges_path = os.path.join(output_dir, "merges_tinystories.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    
    longest_id, longest_bytes = max(vocab.items(), key=lambda x: len(x[1]))
    longest_string = longest_bytes.decode("utf-8", errors="ignore")
    print(f"Longest pretoken ID: {longest_id}, Length: {len(longest_string)}, String: {repr(longest_string)}")