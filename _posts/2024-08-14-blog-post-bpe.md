---
title: 'Transformer Series (1) - Byte-pair Encoding'
date: 2024-08-14
permalink: /posts/2024/08/blog-post-bpe/
excerpt: |
  🐿️ **TL;DR**

  1. Represent arbitrary (Unicode) strings as a sequence of bytes
  2. Train a BPE tokenizer on this byte sequence.
  3. Use this tokenizer to encode text (a string) into tokens (a sequence of integers) for language modeling.
tags:
  - Tokenizer
  - Transformer
  - LLM
toc: true
---

<div class="notice--info" markdown="1">
🐿️ **TL;DR**

1. Represent arbitrary (Unicode) strings as a sequence of bytes
2. Train a BPE tokenizer on this byte sequence. 
3. Use this tokenizer to encode text (a string) into tokens (a sequence of integers) for language modeling.
</div>

# 1. Unicode and UTF-8

## 1.1 Unicode

**Unicode is a text encoding standard that maps characters to integer code points.** 
In Python, we can use the `ord()` function to convert a single Unicode character into its integer representation. The `chr()` function converts an integer Unicode code point into a string with the corresponding character.

```python
>>> ord('田')
30000
>>> chr(30000)
'田'
```

## 1.2 UTF

**Unicode encoding converts a Unicode character into a sequence of bytes.** The Unicode standard itself defines three encodings: UTF-8, UTF-16, and UTF-32, with UTF-8 being the dominant encoding for the Internet (more than 98% of all webpages).

In Python, we can use the `encode()` function to encode a Unicode string into UTF-8. To access the underlying byte values for a Python bytes object, we can iterate over it (e.g., call `list()`). Finally, we can use the `decode()` function to decode a UTF-8 byte string into a Unicode string.

```python
>>> test_string = "hello! 你好!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe4\xbd\xa0\xe5\xa5\xbd!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 228, 189, 160, 229, 165, 189, 33]
>>> print(utf8_encoded.decode("utf-8"))
hello! 你好!

```

## 1.3 Behind Unicode Encoding

UTF-8 encodes each Unicode code point using 1 to 4 bytes, depending on its code point range.

| Code point range | UTF-8 format |
| --- | --- |
| U+0000 to U+007F | 1 byte: `0xxxxxxx` |
| U+0080 to U+07FF | 2 bytes: `110xxxxx 10xxxxxx` |
| U+0800 to U+FFFF | 3 bytes: `1110xxxx 10xxxxxx 10xxxxxx` |
| U+10000 to U+10FFFF | 4 bytes: `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |

**For example,**

Character: **`你`**

Unicode code point: **`U+4F60`**

Binary representation: `0100 1111 0110 0000`

This code point falls into the **3 byte UTF-8 format**: `1110xxxx 10xxxxxx 10xxxxxx`

Fill the bits into the format: `1110 0100` \| `10 111101` \| `10 100000`

Final UTF-8 encoding:  `11100100 10111101 10100000` = `E4 BD A0`

## 1.4 Summary

- **Unicode** (Character Set Standard)

**Character** → **Code Point**. Each character is mapped to a unique number.

- **UTF** (Variable-length Encoding)

**Unicode character → Bytes**. Converts Unicode numbers into binary storage. 

For UTF-8, *English (ASCII): 1 byte; Most Chinese: 3 bytes; Emojis: 4 bytes*

# 2 Subword Tokenization

## 2.1 Word Level

Word level tokenization suffers from the out of vocabulary problem because natural language can keep producing new word types, while the model’s vocabulary is fixed and finite.

## 2.2 Character Level

While the Unicode standard defines a mapping from characters to code points (integers), it’s impractical to train tokenizers directly on Unicode code points, since the **vocabulary would be prohibitively large** (~ 150K items until 2025) and **sparse** (since many characters are quite rare).

## 2.3 Byte Level

Unicode encodings can make vocab 256, which can totally solve OOV problem, but results in extremely long input sequences. The model has to process more tokens, which makes training more difficult.

## 2.4 Subword Level

Subword tokenization is a midpoint between word-level tokenizers and byte-level tokenizers.

Sub level tokenization starts from byte level units, so it avoids the OOV problem. It then merges frequent byte sequences to make input sequences shorter, while making the model trainable.

Subword tokenizers with vocabularies constructed via BPE are often called **BPE tokenizers.** The process of constructing the BPE tokenizer vocabulary is known as “training” the **BPE tokenizer**.

# 3  BPE Tokenizer Training

How do we select these subword units to add to our vocabulary?

R. Sennrich et al. propose to use byte-pair encoding (BPE), a compression algorithm that iteratively replaces (“merges”) the most frequent pair of bytes with a single, new unused index.

The BPE tokenizer training procedure consists of **three main steps**.

## 3.1 Step one: Vocabulary initialization

The tokenizer vocabulary is a one-to-one mapping from bytestring token to integer ID.  Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all bytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.

When encoding text, it’s often desirable to treat some strings as “special tokens” that should never be split into multiple tokens (i.e., will always be preserved as a single token). 

For example, the end-of-sequence string `<|endoftext|>` should always be preserved as a single token (i.e., a single integer ID), so we know when to stop generating from the language model. These special tokens must be added to the vocabulary, so they have a corresponding fixed token ID.

So, complete code is:

```python
vocab = {i: bytes([i]) for i in range(256)}
next_id = 256
for tok in special_tokens:
    vocab[next_id] = tok.encode("utf-8")
    next_id += 1
```

## 3.2 Step two: Pre-tokenization

Once we have a vocabulary, we could, in principle, count how often bytes occur next to each other in your text and begin merging them starting with the most frequent pair of bytes. But!

### **3.2.1 Without pre tokenization**

Treat the **entire corpus** as one long byte string, then count all adjacent byte pairs. 

Problem 1:  Since we’d have to take a full pass over the corpus each time we merge, it’s  computationally expensive.

*For example, suppose the corpus is 100 GB and we need to perform 50,000 merges. If each merge requires scanning the entire 100 GB again, then that would be: ～50,000 × 100 GB*

Problem 2: Adjacent positions are all counted, including between words and punctuation, and between words. This may result in noise tokens.(like “dog.” “dog!”)

So, we use pre tokenization before training. 

### **3.2.2 With pre-tokenization**

Only adjacent byte pairs within the **same pre token** are counted; pairs across pre token boundaries are not counted.

Most modern tokenizers use a regex-based pre-tokenizer, a practice from GPT-2;

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

```python
**Algorithm: Pretokenization**
Input:
  - file_path: Path to the input corpus file
  - special_tokens: List of special tokens (e.g., ["<|endoftext|>"])
  - num_processes: Number of parallel processes

Output:
  - token_freqs: Dictionary where keys are byte sequence tuples and values are frequencies
    Format: {(b'byte1', b'byte2', ...): frequency, ...}
    Example: {(b't', b'h', b'e'): 120, (b'i', b's'): 85, ...}
```

#### 💡 Optimization: Parallelization

First, chunk the corpus while ensuring the chunk boundaries occur at the beginning of a special token. Second, respectively counting with the built-in library multiprocessing.

Code: Please see [here](https://github.com/hanntian/assignment1-basics/blob/06add0d164c797ca1e1b7a84e1a1f54cea669fd2/cs336_basics/pretokenization_example.py#L71)

## **3.3** Step three: Compute BPE merges

Now that we’ve converted our input text into pre-tokens and represented each pre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer).

Compute: the BPE algorithm iteratively counts every pair of bytes and identifies the pair with the highest frequency (“A”, “B”).  When the frequencies are the same, use the **lexicographically largest** option as the tie-breaker.

Merge: Every occurrence of this most frequent pair (“A”, “B”) is then merged, i.e., replaced with a new token “AB”. This new merged token is added to our vocabulary.

```python
**Algorithm: Merge Iteration**

Input:
    -token_freqs: A dictionary mapping token sequences to their frequencies.
                  Example:{% raw %}{{(b'byte1', b'byte2', ...): frequency, ...}{% endraw %}
		-vocab_size: The target vocabulary size.
		-vocab:The initialized vocabulary.
		-next_id:The next available token ID.

Output:
    -vocab: The extended vocabulary.
    -merges: A list of merge operations.
             Example: [(pair1), (pair2), ...]
```

#### 💡 Optimization: Parallelization

Since the only pair counts that change after each merge are those that overlap with the merged pair. Thus, BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. That’s why we maintain `pair_counts= Counter()` in our code. 

Note that the merging part is not parallelizable, since we need best pair in entire token_freqs, and every iteration depends on last time.

Code: Please see [here](https://github.com/hanntian/assignment1-basics/blob/06add0d164c797ca1e1b7a84e1a1f54cea669fd2/tests/adapters.py#L576 Lin)

# 3 BPE Tokenizer: Encoding and Decoding

OK, we finished the BPE training and got a vocab and a merge pair list. 

Now, we will implement a BPE tokenizer that **loads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs**.

## 3.1  Encoding text

The process of encoding text by BPE mirrors how we train the BPE vocabulary. 

- Split input text using user-defined special tokens to multiple parts.
- Pretokenize each part.
- For each pretoken, converted to a sequence of bytes, apply the merge operations in the same order in which they were created, until no more merges can be applied.
- Look up each resulting token in the vocabulary to obtain its corresponding index.
- Repeat this process for all pretokens, and concatenate all token indices into a single list.

Code: Please see [here](https://github.com/hanntian/assignment1-basics/blob/06add0d164c797ca1e1b7a84e1a1f54cea669fd2/cs336_basics/tokenizer.py#L41)

## 3.2 Decoding text

To decode a sequence of integer token IDs back to raw text, we can simply look up each ID’s
corresponding entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a Unicode string.

# 4 Experiments

In this part, I'll show how to train BPE on TinyStories [R. Eldan et al., 2023] dataset  which are single, large plaintext files. Considering the running time, we choose data/TinyStoriesV2-GPT4-valid.txt. And, set `vocab_size = 10,000`，`special_tokens = ["<|endoftext|>"]`. 

Let’s focus on the code efficiency,

## 4.1 Merge Bottleneck

I use profiling tools(cProfile) to identify the bottlenecks.

![profiling result](/images/bpe.png)

**Problem**: The profiler shows that the main bottleneck is inside the merge loop, not file reading or multiprocessing pretokenization. Most time is spent on repeatedly calling `max`, `dict.get`, `len`, which suggests that the implementation scans the entire pair frequency dictionary in every merge iteration. 

To be more specific, the latency lies in which we get the best pair before each merge:

```python
best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
```

**Solution:** To improve performance, we should avoid recomputing the most frequent pair from scratch each time. Instead, we can maintain a dynamic max frequency structure, such as a priority queue, and update only the pairs affected by each merge.

So we can use **max** **heap + lazy deletion (as heap doesn’t support effective random update),** and the core idea is: 
Every time a pair's frequency changes, instead of removing the old record, we directly push a new record in. Later, when retrieving the highest-frequency pair, we check whether the top of the heap is expired. If the record popped from the heap is an old one—for example, the heap shows 120 but `pair_freqs[pair]` is already 80—then it is a stale entry, so we just discard it and keep popping.