import os
import time
import json

from collections import Counter
from difflib import SequenceMatcher

import fasttext
from fasttext import util as ftu
from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt


# Define useful directories
work_dir = os.getcwd()
root_dir = os.path.dirname(work_dir)
sourceccode_dir = os.path.dirname(root_dir)
project_dir = os.path.dirname(sourceccode_dir)
glove_dir = os.path.join(project_dir, '2. Data Acquisition', 'GloVe_pretrained')
fasttext_dir = os.path.join(project_dir, '2. Data Acquisition', 'FastText_pretrained')
stats_dir = os.path.join(work_dir, 'stats')
data_dir = os.path.join(work_dir, 'data', 'hotel')
domain_dir = os.path.join(work_dir, 'domain_embeddings')


def load_FastText(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Cannot find {model_path}')
    print(f"\n\n\nLoading FastText model \n\t{model_path} ...")
    start = time.time()
    fasttext_model = fasttext.load_model(model_path)
    stop = time.time()
    run_time = stop - start
    run_time = '{:.0f}m {:.0f}s'.format(run_time//60, run_time%60)
    print('... in', run_time)
    return fasttext_model


def load_GloVe(model_path: str, n_words: int=500_000):

    # Convert GloVe to word2vec format
    # from gensim.scripts.glove2word2vec import glove2word2vec
    # glove2word2vec(glove_input_file=model_path, 
    #             word2vec_output_file=model_path.replace('.txt', '.gensim'))

    # Load GloVe by Gensim
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Cannot find {model_path}')

    print(f"\n\n\nLoading GloVe model \n\t{model_path} ...")
    start = time.time()
    glove_model = KeyedVectors.load_word2vec_format(model_path, limit=n_words, binary=False)
    # glove_model.syn0norm = glove_model.syn0 # prevent re-calculation of normalized vectors
    glove_vocab = list(glove_model.vocab.keys())
    stop = time.time()
    run_time = stop - start
    run_time = '{:.0f}m {:.0f}s'.format(run_time//60, run_time%60)
    print('... in', run_time)
    return glove_model, glove_vocab


def visualize_word_freqs(word_freqs: list or tuple, process: str='word_freqs', dpi: int=1023):
    for freq_type in ['most', 'least']:
        fig = plt.figure()
        labels, values = zip(*word_freqs)
        probs = np.array(values) / np.sum(values)
        probs = probs.tolist()
        if freq_type == 'most':
            labels, probs = labels[:169], probs[:169]
        else:
            labels, probs = labels[::-1][:69], probs[::-1][:69]
        plt.bar(x=labels, height=probs, width=0.69)
        plt.xticks(fontsize=2, rotation=45)
        fig.savefig(os.path.join(stats_dir, f'{process}_{freq_type}_common.png'), dpi=dpi)


def initialize_vocab(data_dir: str, n_words: int=-1, plot_stats: bool=True) -> dict:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Cannot find {data_dir}')

    source_count = [('<pad>', 0), ('<oov>', 0)]
    source_word2idx = {}
    max_sent_len = 0
    for process in ['train', 'val', 'test']:
        print('Loading {}...'.format(process))
        fname = os.path.join(data_dir, process, 'sentence.txt')

        with open(fname, 'r', encoding='utf-8') as f_reader:
            lines = f_reader.readlines()
            source_words = []
            for line in lines:
                tokens = line.strip().split()
                source_words.extend([tok.lower() for tok in tokens])
                if len(tokens) > max_sent_len:
                    max_sent_len = len(tokens)

        word_freqs = Counter(source_words).items()
        word_freqs = sorted(word_freqs, key=lambda item: item[1])[::-1]
        # word_freqs = list(reversed(word_freqs))

        # word-to-index mapping
        source_count.extend(word_freqs[:n_words])
        for word, _ in source_count:
            if word not in source_word2idx:
                source_word2idx[word] = len(source_word2idx)

        # Visualize
        if plot_stats:
            visualize_word_freqs(word_freqs, process)

    # print(source_count)
    # print(source_word2idx)
    print(f'\t--> max_sentence_length: {max_sent_len}')
    return source_word2idx


def query_glove(glove_model, word: str or list) -> np.array:
    return glove_model.wv.__getitem__(word)


def query_fasttext(fasttext_model, word: str) -> np.array:
    return fasttext_model.get_word_vector(word)


def xavier_initialize(shape_in: list or tuple, shape_out: list or tuple):
    dim_in = np.prod(shape_in)
    dim_out = np.prod(shape_out)
    limit = np.sqrt(6.0/(dim_in+dim_out))
    return np.random.uniform(-limit, limit, size=shape_out)


def extract_ngrams(word, min_n, max_n):
    # Padding
    BoW, EoW = '', ''
    word_extended = BoW + word + EoW

    # Extract
    ngrams = []
    if 2 < len(word) <= 4:
        min_n = 2
    elif len(word) <= 2:
        min_n = 1
    for L in range(min_n, min(len(word_extended), max_n)+1):
        for i in range(len(word_extended)-L+1):
            ngrams += [word_extended[i:i+L]]
    return ngrams


def build_embedding(data_dir: str, word2idx: dict, 
                    glove_vocab: dict, glove_model, 
                    dimension: int=300) -> np.array:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'Cannot find {data_dir}')
    # if dimension > 300:
    #     raise ValueError(f'dimension (={dimension}) cannot be higher than 300')
    # elif dimension < 300:
    #     ftu.reduce_model(fasttext_model, dimension)

    vocab_size = len(word2idx)

    # Matrix Initializer
    w2v = xavier_initialize([vocab_size], [vocab_size, dimension])

    # Reduce GloVe for narrow domain, use FastText for Out-of-Vocab
    oov_list = []
    for word, idx in word2idx.items():
        if word in ['<pad>', '<oov>']:
            continue
        elif word in glove_vocab:
            w2v[idx] = query_glove(glove_model, word)
        else:
            # w2v[idx] = query_fasttext(fasttext_model, word)
            oov_list.append(word)

    # Build ngrams model for Out-of-Vocab
    oov2vec = build_ngram_embeddings(oov_list, dimension, glove_model, glove_vocab)
    for word_ in oov_list:
        w2v[word2idx[word_]] = oov2vec[word_]

    return w2v


def build_ngram_embeddings(word_list: list or tuple, dimension: int, 
                          glove_model, glove_vocab: list or tuple) -> dict:
    # Build dictionary
    word2ngram, ngram2idx = dict(), dict()
    for word in word_list:
        ngrams = extract_ngrams(word, 3, 10)
        word2ngram[word] = ngrams
        for ngram in ngrams:
            if ngram not in ngram2idx:
                ngram2idx[ngram] = len(ngram2idx)

    # Build ngram embeddings
    vocab_size = len(ngram2idx)
    ngram_embeddings = xavier_initialize([vocab_size], [vocab_size, dimension])
    for idx, ngram in ngram2idx.items():
        if ngram in glove_vocab:
            ngram_embeddings[idx] = query_glove(glove_model, ngram)
    
    # Build word embeddings
    word2vec = dict()
    for word, ngrams in word2ngram.items():
        word_vec = xavier_initialize([1], [1, dimension])
        for ngram in ngrams:
            word_vec += ngram_embeddings[ngram2idx[ngram]]
        word_vec /= np.float(len(ngrams)) + 1e-7
        word2vec[word] = word_vec

    return word2vec


def compare_str(a: str, b: str) -> float:
    similarity = SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return float(similarity)
    

if __name__ == "__main__":
    word2idx = initialize_vocab(data_dir)
    with open(os.path.join(data_dir, 'word2id.txt'), 'w', encoding='utf-8') as f_writer:
        f_writer.write(json.dumps(word2idx, indent=4))

    # Load pretrained word2vec
    glove_model, glove_vocab = load_GloVe(os.path.join(glove_dir, 'glove.840B.300d.gensim'))
    domain_model, domain_vocab = load_GloVe(os.path.join(domain_dir, 'restaurant.gensim'))
    # fasttext_model = load_FastText(os.path.join(fasttext_dir, 'cc.en.300.bin'))

    # Build word2vec for narrow domain
    w2v_300 = build_embedding(data_dir, word2idx, glove_vocab, glove_model, 300)
    w2v_100 = build_embedding(data_dir, word2idx, domain_vocab, domain_model, 100)

    np.save(os.path.join(data_dir, 'global_embeddings.npy'), w2v_300)
    np.save(os.path.join(data_dir, 'domain_embeddings.npy'), w2v_100)





