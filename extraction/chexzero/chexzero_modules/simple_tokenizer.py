"""
MIT License

Copyright (c) 2021 OpenAI
Modifications made to align with specific project usage.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import gzip
import html
import os
from functools import lru_cache

# import ftfy  # Optional dependency, commented out to match usage in probe_multimodal.py
import regex as re


@lru_cache()
def default_bpe():
    """
    Returns the default path to the BPE vocabulary file.
    Checks the script's directory first, then the user's CLIP cache.

    Returns:
        str: The path to the BPE vocabulary file.

    Raises:
        FileNotFoundError: If the BPE file cannot be found in expected locations.
    """
    try:
        # Try finding BPE vocab next to the script first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bpe_path = os.path.join(script_dir, "bpe_simple_vocab_16e6.txt.gz")
        if os.path.exists(bpe_path):
            return bpe_path
        else:
            # If not found, check the standard CLIP cache location
            bpe_path_alt = os.path.expanduser("~/.cache/clip/bpe_simple_vocab_16e6.txt.gz")
            if os.path.exists(bpe_path_alt):
                # print(f"BPE vocab not found at {bpe_path}, using cache: {bpe_path_alt}") # Optional logging
                return bpe_path_alt
            else:
                # If not found in either location, raise an error
                raise FileNotFoundError(
                    f"BPE Vocab file not found at expected locations: {bpe_path} or {bpe_path_alt}"
                )
    except Exception as e:
         # Catch potential permission errors with expanduser etc.
         raise FileNotFoundError(f"Could not access BPE Vocab file locations: {e}")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    if len(word) > 1: # Check length to avoid index error on single-char words
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
    return pairs


def basic_clean(text):
    """
    Performs basic text cleaning: HTML unescaping and stripping whitespace.
    NOTE: ftfy fix is commented out to match usage in probe_multimodal.py.
    """
    # text = ftfy.fix_text(text) # Optional: fix unicode errors, requires ftfy library
    text = html.unescape(html.unescape(text)) # Double unescape for nested entities
    return text.strip()


def whitespace_clean(text):
    """
    Removes extra whitespace and strips leading/trailing whitespace.
    """
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    """
    A simple BPE Tokenizer based on OpenAI's implementation, modified for specific project needs.
    """
    def __init__(self, bpe_path: str = default_bpe()):
        """
        Initializes the tokenizer.

        Args:
            bpe_path (str): Path to the BPE vocabulary file (gzipped).
                             Defaults to checking script dir then ~/.cache/clip/.
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1] # Skip header, take relevant merges
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab] # Add end-of-word tokens
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>']) # Add special tokens

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # Cache common tokens and BPE results
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        # Regex pattern for token splitting
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        """
        Applies Byte Pair Encoding to a single token using cached ranks.

        Args:
            token (str): The token (unicode string) to encode.

        Returns:
            str: The BPE-encoded token string (space-separated subwords).
        """
        if token in self.cache:
            return self.cache[token]

        # Represent word as tuple of characters, add end-of-word marker
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>' # Return token with end-of-word if no pairs

        while True:
            # Find the lowest rank pair based on learned merges
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break # No more mergeable pairs found

            # Merge the best pair
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # Find the first occurrence of 'first' starting from index i
                    j = word.index(first, i)
                    # Add characters before the found occurrence
                    new_word.extend(word[i:j])
                    i = j # Move index to the start of the found occurrence
                except ValueError: # More specific exception handling
                    # 'first' not found in the rest of the word
                    new_word.extend(word[i:])
                    break # Exit the inner loop

                # Check if the next symbol is 'second' to perform the merge
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second) # Append the merged bigram
                    i += 2 # Advance index past the merged pair
                else:
                    new_word.append(word[i]) # Append the current character
                    i += 1 # Advance index by one

            # Update word and pairs for the next iteration
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break # Word is fully merged into a single unit
            else:
                pairs = get_pairs(word) # Recalculate pairs for the new word structure

        # Join the final subword units with spaces
        word = ' '.join(word)
        self.cache[token] = word # Cache the BPE result for this token
        return word

    def encode(self, text):
        """
        Encodes a string into a list of BPE token IDs.
        Handles potential encoding errors and skips problematic tokens.

        Args:
            text (str): The input string.

        Returns:
            list[int]: A list of integer token IDs.
        """
        bpe_tokens = []
        # Only apply whitespace clean and lowercasing, skipping basic_clean (ftfy)
        text = whitespace_clean(text).lower()
        # Tokenize using the regex pattern
        for token in re.findall(self.pat, text):
            try:
                # Convert token to bytes (replacing errors) -> unicode characters using byte_encoder map
                token_bytes = token.encode('utf-8', errors='replace')
                token_unicode = ''.join(self.byte_encoder[b] for b in token_bytes)

                # Apply BPE to the unicode token and get integer IDs from the encoder map
                bpe_encoded_token = self.bpe(token_unicode)
                bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in bpe_encoded_token.split(' '))
            except Exception as e:
                # Skip tokens that cause errors during encoding or BPE
                # print(f"Warning: Skipping token '{token}' due to BPE/Encoding error: {e}") # Optional: for debugging
                continue # Move to the next token
        return bpe_tokens

    def decode(self, tokens):
        """
        Decodes a list of BPE token IDs back into a string.

        Args:
            tokens (list[int]): List of integer token IDs.

        Returns:
            str: The decoded string.
        """
        # Map IDs to BPE tokens (unicode strings) using the decoder map
        text = ''.join([self.decoder[token] for token in tokens])
        # Map unicode characters back to bytes using byte_decoder -> decode utf-8 (replacing errors)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace")
        # Replace end-of-word markers with spaces for readability
        text = text.replace('</w>', ' ')
        return text

# Example usage (optional, can be commented out or placed under if __name__ == "__main__":)
# if __name__ == '__main__':
#     tokenizer = SimpleTokenizer()
#     text_example = "This is a <|startoftext|> test sentence! 123."
#     encoded = tokenizer.encode(text_example)
#     print(f"Original: {text_example}")
#     print(f"Encoded: {encoded}")
#     decoded = tokenizer.decode(encoded)
#     print(f"Decoded: {decoded}")