class BPETokenizer:
    def __init__(self):
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        return

"""
SAMPLE_FILE = '/Users/jon/Desktop/llm-from-scratch/data/file1.txt'
EOW = "W" # the end of word symbol

Byte-pair encoding
- begin with a set of individual characters
- next choose the two tokens that are most frequently adjacent
- merge these tokens together and add them to the vocabulary
- replace every (token1, token2) with new_token in the vocabulary

def append_word_boundaries(corpus):
    new_corpus = ''
    for word in corpus:
        new_corpus += word
        new_corpus += EOW
    return new_corpus

def initial_vocab(corpus):
    vocab = Counter()
    corpus = list(corpus)

    for c in range(len(corpus) - 1):
        if corpus[c] == EOW:
            continue

        token = [ corpus[c], corpus[c+1] ]
        vocab[tuple(token)] += 1

    return vocab

def merge_vocab(pair, vocab):
    new_token = ''.join(pair)

    vocab

    for token_tuple, freq in vocab.items():
        word_str = ''.join(token_tuple)
        if new_token in word_str:
            # adjust vocab
            pass

def bpe(corpus, max_merges = 10):
    vocab = initial_vocab(corpus)

    for i in range(max_merges):
        most_freq_bigram = max(vocab, key=vocab.get)
        merge_vocab(most_freq_bigram, vocab)

def load_data(filename):
    with open(filename, 'r') as f:
        text = f.read()
    words = text.split()
    return words

#################################
# TESTING
#################################

data = load_data(SAMPLE_FILE)
corpus = append_word_boundaries(data)
print(bpe(corpus))
"""