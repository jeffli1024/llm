"""
BPE (Byte Pair Encoding) Tokenizer - A step-by-step implementation for learning.

BPE works by:
1. Start with character-level vocabulary (each char = 1 token)
2. Find the most frequent pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat until you reach target vocabulary size

This gives subword tokens: common words stay whole, rare words split into pieces.
"""

from collections import defaultdict
import re

import requests


def get_pairs(word: list[str]) -> dict[tuple[str, str], int]:
    """
    Count how often each adjacent pair of tokens appears in a word.
    Example: ["h", "e", "l", "l", "o"] -> {("h","e"):1, ("e","l"):1, ("l","l"):1, ("l","o"):1}
    """
    pairs = defaultdict(int)
    for i in range(len(word) - 1):
        pair = (word[i], word[i + 1])
        pairs[pair] += 1
    return dict(pairs)


def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    """
    Merge all occurrences of the pair into a single token.
    Example: merge_pair(["h","e","l","l","o"], ("l","l")) -> ["h","e","ll","o"]
    """
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            new_tokens.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def train_bpe(
    corpus: str,
    vocab_size: int = 256,
    special_tokens: list[str] | None = None,
) -> dict:
    """
    Train BPE on a text corpus.

    Args:
        corpus: Raw text to learn from
        vocab_size: Target number of tokens (including special tokens)
        special_tokens: Tokens like <pad>, <unk>, <eos> that are always in vocab

    Returns:
        Dict with 'merges' (list of merged pairs) and 'vocab' (token -> id mapping)
    """
    # Step 1: Preprocess - split into words, add word boundary
    # We use </w> to mark end of word so "cat" and "cats" don't merge incorrectly
    words = re.findall(r"\S+|\s+", corpus)  # Split on whitespace, keep spaces
    word_freqs = defaultdict(int)
    for w in words:
        word_freqs[" ".join(list(w)) + " </w>"] += 1

    # Step 2: Initialize vocab with characters + special tokens
    vocab = set()
    for word in word_freqs:
        for char in word.split():
            vocab.add(char)
    if special_tokens:
        vocab.update(special_tokens)

    # Step 3: Convert each word to list of tokens (chars initially)
    splits = {word: word.split() for word in word_freqs}

    # Step 4: Iteratively merge most frequent pairs
    merges = []
    while len(vocab) < vocab_size:
        # Count all pairs across the corpus
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = splits[word]
            for pair, count in get_pairs(tokens).items():
                pair_counts[pair] += count * freq

        if not pair_counts:
            break

        # Pick the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)

        # Add merged token to vocab
        new_token = best_pair[0] + best_pair[1]
        vocab.add(new_token)

        # Update all words: merge this pair everywhere
        for word in word_freqs:
            splits[word] = merge_pair(splits[word], best_pair)

    # Build final vocab: token -> id
    token_to_id = {t: i for i, t in enumerate(sorted(vocab))}
    return {"merges": merges, "vocab": token_to_id}


def tokenize(text: str, merges: list[tuple[str, str]], vocab: dict[str, int]) -> list[int]:
    """
    Tokenize text using learned BPE merges.

    Process: split into chars, then apply merges in the same order they were learned.
    """
    if not text.strip():
        return []

    # Split into words (with </w> marker)
    words = re.findall(r"\S+|\s+", text)
    all_ids = []

    for word in words:
        tokens = list(word) + ["</w>"]
        # Apply each merge in order
        for pair in merges:
            tokens = merge_pair(tokens, pair)
        # Convert to IDs
        for t in tokens:
            if t in vocab:
                all_ids.append(vocab[t])
            else:
                # Fallback: use char-by-char for unknown tokens
                for c in t:
                    all_ids.append(vocab.get(c, vocab.get("<unk>", 0)))
    return all_ids


def detokenize(ids: list[int], id_to_token: dict[int, str]) -> str:
    """Convert token IDs back to text (simple concatenation)."""
    return "".join(id_to_token.get(i, "") for i in ids).replace("</w>", " ")


# =============================================================================
# Example usage & learning exercises
# =============================================================================

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    corpus = requests.get(url).text

    # Tiny corpus for demonstration (alternative)
    # corpus = """
    # the cat sat on the mat
    # the dog ran on the rug
    # cats and dogs are pets
    # """ * 10  # Repeat to get meaningful frequencies

    print("Training BPE (vocab_size=50)...")
    result = train_bpe(corpus, vocab_size=50)
    merges = result["merges"]
    vocab = result["vocab"]
    id_to_token = {v: k for k, v in vocab.items()}

    print(f"\nLearned {len(merges)} merges. First 10:")
    for i, (a, b) in enumerate(merges[:10]):
        print(f"  {i+1}. '{a}' + '{b}' -> '{a+b}'")

    # Test tokenization
    test = "the cats sat"
    ids = tokenize(test, merges, vocab)
    print(f"\n'the cats sat' -> {ids}")
    print(f"Decoded: '{detokenize(ids, id_to_token)}'")
