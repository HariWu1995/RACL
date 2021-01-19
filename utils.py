import os
import time

import re
import json
from string import punctuation
from nltk import sent_tokenize

from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt


################
# For TRAINING #
################

def load_fasttext_model(path, emb_dim: int=-1):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} doesnt exist')
    
    import fasttext as ft

    print('Loading FastText model ...')
    start = time.time()
    model = ft.load_model(path)
    end = time.time()
    print(f'\t... in {int(end-start)}s')
    
    if emb_dim > 0:
        if emb_dim < model.get_dimension():
            import fasttext.util as ftu

            print('Reducing dimension of FastText model ...')
            start = time.time()
            ftu.reduc_model(model, emb_dim)
            end = time.time()
            print(f'\t... in {int(end-start)}s')
    return model


def read_data(fname: str, source_word2idx, max_length: int, 
              is_testing: bool=False, feed_embedding: bool=False):
    source_data, source_mask, aspect_y, opinion_y, sentiment_y, sentiment_mask, position_m = [], [], [], [], [], [], []

    reviews = open(os.path.join(fname, 'sentence.txt'), 'r', encoding='utf-8').readlines()
    ae_data = open(os.path.join(fname, 'target.txt'), 'r', encoding='utf-8').readlines()
    oe_data = open(os.path.join(fname, 'opinion.txt'), 'r', encoding='utf-8').readlines()
    sa_data = open(os.path.join(fname, 'target_polarity.txt'), 'r', encoding='utf-8').readlines()
    for index in range(len(reviews)):

        # Read Sentence Input
        idx, mask = [], []
        len_cnt = 0
        sptoks = reviews[index].strip().split()
        for sptok in sptoks:
            if len_cnt >= max_length:
                break
            sptok = sptok.lower()
            if not feed_embedding:
                if sptok not in source_word2idx:
                    sptok = '<oov>'
            idx.append(source_word2idx[sptok] if not feed_embedding else source_word2idx.get_word_vector(sptok))
            mask.append(1.)
            len_cnt += 1

        n_pads = max_length - len(idx)
        if not feed_embedding:
            source_data.append(idx + [0]*n_pads)
        else:
            emb_dim = source_word2idx.get_dimension()
            source_data.append(idx + [[0]*emb_dim]*n_pads)
        source_mask.append(mask + [0.]*n_pads)

        # Read Aspect labels
        ae_labels = ae_data[index].strip().split()
        aspect_label = [
            tf.keras.utils.to_categorical(int(l), num_classes=3, dtype=int).tolist() for l in ae_labels
        ]

        # Read Opinion labels
        oe_labels = oe_data[index].strip().split()
        opinion_label = [
            tf.keras.utils.to_categorical(int(l), num_classes=3, dtype=int).tolist() for l in oe_labels
        ]

        # Read Sentiment labels
        sa_labels = sa_data[index].strip().split()
        sentiment_label, sentiment_indi = [], []
        for l in sa_labels:
            l = int(l)
            if 1 <= l <= 3:
                sentiment_label.append(tf.keras.utils.to_categorical(l-1, num_classes=3, dtype=int).tolist())
                sentiment_indi.append(1.)
            else:
                sentiment_label.append([0, 0, 0])
                sentiment_indi.append(1. if is_testing else 0.)
                
        # 0-padding for consistent length of input
        aspect_y.append(aspect_label + [[0, 0, 0]]*n_pads)
        opinion_y.append(opinion_label + [[0, 0, 0]]*n_pads)
        sentiment_y.append(sentiment_label + [[0, 0, 0]]*n_pads)
        position_m.append(position_matrix(len(idx), max_length))
        sentiment_mask.append(sentiment_indi + [0.]*n_pads)

    # Ensure correct input format
    aspect_y = [np.array(ay[:max_length]).astype(np.int) for ay in aspect_y]
    opinion_y = [np.array(oy[:max_length]).astype(np.int) for oy in opinion_y]
    sentiment_y = [np.array(sa[:max_length]).astype(np.int) for sa in sentiment_y]
    source_mask = [np.array(xm[:max_length]).astype(np.float) for xm in source_mask]
    sentiment_mask = [np.array(sm[:max_length]).astype(np.float) for sm in sentiment_mask]

    return np.array(source_data, dtype=int), np.array(aspect_y, dtype=int), \
           np.array(opinion_y, dtype=int), np.array(sentiment_y, dtype=int), \
           np.array(source_mask, dtype=float), np.array(sentiment_mask, dtype=float), \
           np.array(position_m, dtype=float)


def position_matrix(sen_len: int, max_len: int) -> np.array:
    A = np.zeros([max_len, max_len], dtype=np.float32)
    for i in range(sen_len):
        for j in range(sen_len):
            if i == j:
                A[i][j] = 0.
            else:
                distance = abs(i-j)
                A[i][j] = 1 / (np.log2(2+distance))
                # A[i][j] = 1 / (abs(i-j))
    return A


def count_parameter(trainable_variables):
    # total_parameters = 0
    # for variable in trainable_variables:
    #     shape = variable.get_shape()
    #     print(variable, shape)
    #     variable_parameters = 1
    #     for dim in shape:
    #         variable_parameters *= dim.value
    #     total_parameters += variable_parameters
    total_parameters = np.sum(
        [np.prod(var.get_shape().as_list()) for var in trainable_variables]
    )
    return (total_parameters)


def min_max_normal(tensor):
    dim = tf.shape(tensor)[-1]
    max_value = tf.reduce_max(tensor, -1, keepdims=True)
    max_value = tf.tile(max_value, [1, 1, dim])
    min_value = tf.reduce_min(tensor, -1, keepdims=True)
    min_value = tf.tile(min_value, [1, 1, dim])
    norm_tensor = (tensor-min_value) / (max_value-min_value+1e-6)
    return norm_tensor


def z_score_normal(tensor):
    dim = tf.shape(tensor)[-1]
    axes = [2]
    mean, variance = tf.nn.moments(tensor, axes, keep_dims=True)
    std = tf.sqrt(variance)
    mean = tf.tile(mean, [1, 1, dim])
    std = tf.tile(std, [1, 1, dim])
    norm_tensor = (tensor - mean) / (std + 1e-6)
    return norm_tensor


def smooth_labels(labels, factor=0.1):
    # smooth the labels
    n_classes = labels.shape[-1]
    weights = 1. - factor
    bias = factor / n_classes
    labels = labels*weights + bias
    return labels


def plot_history(history, figsize=(6, 9), return_figure: bool=True, **kwargs):
    """
    Plot the training history of one or more models.
    This creates a column of plots, with one plot for each metric recorded during training, with the
    plot showing the metric vs. epoch. If multiple models have been trained (that is, a list of
    histories is passed in), each metric plot includes multiple train and validation series.
    Validation data is optional (it is detected by metrics with names starting with ``val_``).
    
    Args:
        history: the training history, as returned by :meth:`tf.keras.Model.fit`
        individual_figsize (tuple of numbers): the size of the plot for each metric
        return_figure (bool): if True, then the figure object with the plots is returned, None otherwise.
        kwargs: additional arguments to pass to :meth:`matplotlib.pyplot.subplots`
    
    Returns:
        :class:`matplotlib.figure.Figure`: The figure object with the plots if ``return_figure=True``, None otherwise
    
    Reference:
        https://github.com/stellargraph/stellargraph/blob/develop/stellargraph/utils/history.py
    """

    # explicit colours are needed if there's multiple train or multiple validation series, because
    # each train series should have the same color. This uses the global matplotlib defaults that
    # would be used for a single train and validation series.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_train = colors[0]
    color_validation = colors[1]

    if not isinstance(history, list):
        history = [history]

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix) :]

    metrics = sorted({remove_prefix(m, "val_") for m in history[0].history.keys()})

    height, width = figsize
    overall_figsize = (width, len(metrics)*height)

    # plot each metric in a column, so that epochs are aligned (squeeze=False, so we don't have to
    # special case len(metrics) == 1 in the zip)
    fig, all_axes = plt.subplots(
        len(metrics), 1, squeeze=False, sharex="col", figsize=overall_figsize, **kwargs
    )

    has_validation = False
    for ax, m in zip(all_axes[:,0], metrics):
        for h in history:
            # summarize history for metric m
            ax.plot(h.history[m], c=color_train)

            try:
                val = h.history["val_" + m]
            except KeyError:
                # no validation data for this metric
                pass
            else:
                ax.plot(val, c=color_validation)
                has_validation = True

        ax.set_ylabel(m, fontsize="x-large")

    # don't be redundant: only include legend on the top plot
    labels = ["train"]
    if has_validation:
        labels.append("validation")
    all_axes[0, 0].legend(labels, loc="best", fontsize="x-large")

    # ... and only label "epoch" on the bottom
    all_axes[-1, 0].set_xlabel("epoch", fontsize="x-large")

    # minimise whitespace
    fig.tight_layout()

    if return_figure:
        return fig


def debug(x):
    print(f'\n\n\n{x}\n\n\n')



#################
# For INFERENCE #
#################

# Define useful variables
word_pattern = re.compile(r'\w+')
punct_pattern = re.compile(f"[{punctuation}]")
sentiment_dict = {
    0: 'positive',
    1: 'negative',
    2: 'neutral',
    # 4: 'conflict'
    # 0: 'background',
}
term_dict = {
    1: 'Beginning',
    2: 'Inside',
    0: 'Outside',
}


def split_words(text: str) -> list:
    # print(f"\n{text}")

    # Split sentence into phrases by punctuation
    punct_locs = [(p.start(), p.end()) for p in punct_pattern.finditer(text)]
    if len(punct_locs)==0:
        phrases_dict = {0: [0, len(text), text]}
    else:
        phrases_dict = dict()
        phrase_idx = 0
        last_punct_end = 0
        for p_i, punct_loc in enumerate(punct_locs):
            current_punct_start, current_punct_end = punct_loc
            if p_i == 0:
                if current_punct_start > 0:
                    phrases_dict[phrase_idx] = [0, current_punct_start, text[:current_punct_start]]
                    phrase_idx += 1
            elif p_i != 0:
                phrases_dict[phrase_idx] = [last_punct_end, current_punct_start, text[last_punct_end:current_punct_start]]
                phrase_idx += 1
            phrases_dict[phrase_idx] = [current_punct_start, current_punct_end, text[current_punct_start:current_punct_end]]
            phrase_idx += 1
            if p_i == len(punct_locs)-1:
                if current_punct_end < len(text):
                    phrases_dict[phrase_idx] = [current_punct_end, len(text)-1, text[current_punct_end:]]
            last_punct_end = current_punct_end

    # Split phrases into words (offset by sentence, not by current phrase)
    words_dict = dict()
    word_idx = 0
    for phrase_idx in range(len(phrases_dict)):
        phrase_start, phrase_end, phrase = phrases_dict[phrase_idx]
        if phrase_end-phrase_start == 1: # case of punctuation
            words_dict[word_idx] = phrases_dict[phrase_idx]
            word_idx += 1    
        phrase_words_dict = {
            w_i+word_idx: [w.start()+phrase_start, w.end()+phrase_start, w.group(0)] \
                for w_i, w in enumerate(word_pattern.finditer(phrase))
        }
        word_idx += len(phrase_words_dict)
        words_dict.update(phrase_words_dict)

    return [word_data[2] for word_data in words_dict.values()]


def tokenize(sentences: list, 
             word2idx: dict, 
             max_length: int, 
             oov_token: str='<oov>'):
    if not isinstance(sentences, list):
        sentences = [sentences]

    data, masks, position_matrices, original_data = [], [], [], []
    for sentence in sentences:
        # print(sentence)

        tokens = split_words(sentence.lower())
        # print(tokens)
        indices = []
        mask = []
        len_cnt = 0
        for token in tokens:
            if len_cnt >= max_length:
                break
            indices.append(word2idx[token] if token in word2idx else word2idx[oov_token])
            mask.append(1.)
            len_cnt += 1
        n_pads = max_length - len(indices)
        data.append(indices + [0]*n_pads)
        masks.append(mask + [0.]*n_pads)
        position_matrices.append(position_matrix(len(indices), max_length))
        original_data.append(tokens[:max_length])
    return np.array(data, dtype=int), np.array(masks), np.array(position_matrices), original_data


# get html element
def color_str(s: str, color: str='black'):
    if s == ' ':
        return "<text style=color:#000000;padding-left:10px;background-color:{}> </text>".format(color)
    else:
        return "<text style=color:#000000;background-color:{}>{} </text>".format(color, s)
        # return "<u><font color={}><text style=color:#000>{} </text></u>".format(color, s)


def rgb2hex(r: int, g: int, b: int) -> str:
    r = max(min(r, 255), 0)
    g = max(min(g, 255), 0)
    b = max(min(b, 255), 0)
    return f"#{r:02x}{g:02x}{b:02x}" 


def hex2rgb(hex: str) -> list:
    return list(map(ord, hex[1:].decode('hex')))

    
# print html
def print_color(t: list or tuple) -> str:
    # from IPython.display import display, HTML as html_print
    # display(html_print(''.join([color_str(str(ti), color=ci) for ti, ci in t])))
    return ''.join([color_str(str(t_i), color=c_i) for t_i, c_i in t])


def scale_value(value: int, max_value: int=255, min_value: int=200, reverse_value: bool=True) -> int:
    if reverse_value:
        value = 1 - value
    value = max(min(value, 1.0), 0.0)
    value = value * (max_value-min_value) + min_value
    return int(value)


# get appropriate color for value
def get_color(value: float, label: str='background') -> str:

    # assign color code for background
    color_code = "#FFFFFF" 
    
    # if not background, overwrite color code
    if label == 'opinion':
        value = scale_value(value, 200, 128)
        color_code = rgb2hex(value, value, value) # gray
    elif 'aspect' in label:
        value = scale_value(value)
        if 'pos' in label: 
            color_code = rgb2hex(0, value, 0) # green
        elif 'neg' in label:
            color_code = rgb2hex(value, 0, 0) # red
        elif 'neu' in label:
            color_code = rgb2hex(value, value, 0) # yellow
    return color_code


def visualize(results_list: list or tuple) -> str:
    text_colors = []
    for t, v, l in results_list:
        # print(t, v, l)
        text_color = [t, get_color(v, l)]
        # print(f'\t{text_color[1]}')
        text_colors += [text_color]
    return print_color(text_colors)


def split_documents(documents: list or tuple) -> (list, dict):
    # Split documents into multiple single-sentences 
    #       and record the reversed mapping
    sent2doc = dict()
    sentences = []
    for doc_i, doc in enumerate(documents):
        sentences_in_doc = sent_tokenize(doc)
        sentences += sentences_in_doc
        for _ in sentences_in_doc:
            sent2doc[len(sent2doc)] = doc_i
    return sentences, sent2doc


def decode_results(original_sentences, sent2doc, sentences_input, 
                   aspect_probs, opinion_probs, sentiment_probs) -> pd.DataFrame:
    results_df = pd.DataFrame(
        columns=['doc_id', 'word', 'aspect', 'aspect_prob', 'opinion', 'opinion_prob', 'sentiment', 'sentiment_prob']
    )
    r_id = 0
    for sentence_id, (tokens, aspect, opinion, sentiment) in \
        enumerate(zip(sentences_input, aspect_probs, opinion_probs, sentiment_probs)):
        # tokens = list(tokens)
        # n_tokens = tokens.index(0) if 0 in tokens else len(tokens)
        # tokens = tokens[:n_tokens]
        # words = [idx2word[token] for token in tokens]
        words = original_sentences[sentence_id]
        for w_idx, word in enumerate(words):
            results_df.loc[r_id] = [
                sent2doc[sentence_id], word,
                term_dict[np.argmax(aspect[w_idx])], np.max(aspect[w_idx]),
                term_dict[np.argmax(opinion[w_idx])], np.max(opinion[w_idx]),
                sentiment_dict[np.argmax(sentiment[w_idx])], np.max(sentiment[w_idx]),
            ]
            r_id += 1
    return results_df


def write_log(results_df, fn: str, threshold: float=0.5):
    logger = open(fn, 'w')
    for doc_id, doc_df in results_df.groupby(by=['doc_id']):
        # print(doc_df)
        # doc_df.reset_index(inplace=True)
        doc_results = doc_df.values.T.tolist()
        _, words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs = doc_results
        logger.write(f'\n\n\n{"-"*29}\nSentence {doc_id+1:02d}: {len(words)} words - {" ".join(words)}\n')
        for w_idx, (word, aspect, aspect_prob, opinion, opinion_prob, sentiment, sentiment_prob) \
            in enumerate(zip(words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs)):
            logger.write(f'\tWord {w_idx+1:02d}: {word}\n')
            logger.write(f'\t\t{"Aspect":>10s}: {aspect if aspect_prob > threshold else None} - {max(aspect_prob, threshold):.3f}\n')
            logger.write(f'\t\t{"Opinion":>10s}: {opinion if opinion_prob > threshold else None} - {max(opinion_prob, threshold):.3f}\n')
            logger.write(f'\t\t{"Sentiment":>10s}: {sentiment if sentiment_prob > threshold else None} - {max(sentiment_prob, threshold):.3f}\n')
    logger.close()


def write_html(results_df: pd.DataFrame, fn: str):
    n_docs = len(results_df.doc_id.unique())
    logger = open(fn, 'w')
    for doc_id, doc_df in results_df.groupby(by=['doc_id']):
        text_head = f"{'<br>'*5}[{doc_id+1:02d}/{n_docs:02d}]{'<br>'*1}"
        doc_results = doc_df.values.T.tolist()

        results = [] # list of tuples <word, confidence_score, label>
        _, words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs = doc_results
        for word, aspect, aspect_prob, opinion, opinion_prob, sentiment, sentiment_prob \
            in zip(words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs):
            result = [word]
            if aspect != 'Outside':
                result.extend([sentiment_prob*aspect_prob, f'aspect_{sentiment}'])
            elif opinion != 'Outside':
                result.extend([opinion_prob, 'opinion'])
            else:
                result.extend([0.0, 'background'])

            # print(result)
            results.append(result)
        
        text_body = visualize(results)
        logger.write(text_head+text_body)
    logger.close()
    return True


def dict2html(doc_results: dict, fn: str):
    doc_results = doc_results['aspect-based sentiment']
    logger = open(fn, 'w')
    n_docs = len(doc_results)
    for doc_id, doc_result in enumerate(doc_results):
        results_list = list(zip(doc_result['tokens'], doc_result['colors']))
        text_head = f"{'<br>'*5}[{doc_id+1:02d}/{n_docs:02d}]{'<br>'*1}"
        text_body = ''.join([color_str(str(t_i), color=c_i) for t_i, c_i in results_list])
        logger.write(text_head+text_body)
    logger.close()


def format_results(results_df) -> dict:

    doc_results = []
    for doc_id, doc_df in results_df.groupby(by=['doc_id']):
        results = doc_df.values.T.tolist()
        colors, labels = [], []

        _, words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs = results
        for word, aspect, aspect_prob, opinion, opinion_prob, sentiment, sentiment_prob \
            in zip(words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs):
            if aspect != 'Outside':
                label = f'aspect_{sentiment}'
                colors.append(get_color(sentiment_prob*aspect_prob, label))
                labels.append(label)
            elif opinion != 'Outside':
                label = 'opinion'
                colors.append(get_color(opinion_prob, label))
                labels.append(label)
            else:
                colors.append("#FFFFFF")
                labels.append('background')

        doc_results.append({
            'id': doc_id,
            'tokens': words,
            'colors': colors,
            'labels': labels,
        })

    return {'aspect-based sentiment': doc_results}














