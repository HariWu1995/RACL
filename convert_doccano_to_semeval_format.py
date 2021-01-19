import os
import ntpath
import itertools

from glob import glob
from tqdm import tqdm

import re
import spacy
import json
from string import punctuation
from pprint import PrettyPrinter


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Download SpaCy models if needed
spacy_model = 'en_core_web_sm'
try:
    nlp = spacy.load(spacy_model)
except OSError:
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model)


# Define useful directories
working_dir = os.getcwd()
root_dir = os.path.dirname(working_dir)
project_dir = os.path.dirname(os.path.dirname(root_dir))
data_dir = os.path.join(project_dir, '2. Data Acquisition', 'Raw Dataset')
label_dir = os.path.join(data_dir, 'labels')
annotation_dir = os.path.join(data_dir, 'annotation_category', 'extend_4')


# Define useful variables
word_pattern = re.compile(r'\w+')
punct_pattern = re.compile(f"[{punctuation}]")
printer = PrettyPrinter(indent=4)

sentiment_mapper = {
    "pos": 1,
    "neg": 2,
    "neu": 3,
    "positive": 1,
    "negative": 2,
    "neutral": 3,
    "conflict": 4,
}
train_val_test_ratio = [0.6, 0.2, 0.2]
random_seed = 4_10_20


def encode_words_location(text: str) -> dict:
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
    # printer.pprint(phrases_dict)

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
    # printer.pprint(words_dict)

    # Convert word dictionary to word dataframe --> easy comparison
    words_df = pd.DataFrame(data=words_dict).T
    words_df.rename(columns={
        0: 'offset_start',
        1: 'offset_end',
        2: 'word'
    }, inplace=True)
    # print(words_df)
    
    # Sentencize
    words_df['sentence_id'] = [0] * len(words_df)
    sentences = [sent.text for sent in nlp(text).sents]
    BoSs = [text.find(sent) for sent in sentences]
    for sentence_id, bos in enumerate(BoSs):
        if sentence_id == 0:
            continue
        words_df.loc[words_df.offset_start>=bos, 'sentence_id'] = sentence_id

    return words_dict, words_df


def decode_words_location(located_words: pd.DataFrame, 
                          annotations: list or tuple) -> list:
    annotations = list(annotations)
    n_words = len(located_words)
    located_words['word'] = located_words['word'].apply(lambda x: x.lower())

    # Assign all words as BACKGROUND
    located_words['aspect'] = [0] * n_words
    located_words['opinion'] = [0] * n_words
    located_words['sentiment'] = [0] * n_words

    # If no annotation, all words are considered background - class=0
    if len(annotations) < 1:
        return located_words

    for annotation in annotations:
        offset_start, offset_end, label = annotation[:3]

        # Assign all words in annotation as 
        #       BEGINNING for Aspect & Opinion, 
        #       polarity code for Sentiment
        annotation_locs = (located_words['offset_start']>=offset_start) & (located_words['offset_end']<=offset_end)
        if label == 'opinion':
            located_words.loc[annotation_locs, 'opinion'] = 1
        elif label.startswith('aspect'):
            polarity = label.split('_')[1]
            located_words.loc[annotation_locs, 'aspect'] = 1
            located_words.loc[annotation_locs, 'sentiment'] = sentiment_mapper[polarity]

        # Split Aspect & Opinion annotated words into BEGINNING and INSIDE
        for r_i in range(n_words-1, 0, -1):
            for col in ['opinion', 'aspect']:
                if located_words.loc[r_i, col] == 0:
                    continue
                if located_words.loc[r_i-1, col] == 1:
                    # if previous word is annotated as BEGINNING, flip current word to INSIDE
                    located_words.loc[r_i, col] = 2
    # print(located_words)
    return located_words


def window_slide(seq: list or tuple, window_size: int=2):
    seq_iter = iter(seq)
    seq_sliced = list(itertools.islice(seq_iter, window_size))
    if len(seq_sliced) == window_size:
        yield seq_sliced
    for seq_el in seq_iter:
        yield seq_sliced[1:] + [seq_el]


if __name__ == "__main__":

    labels = ['sentence', 'target', 'opinion', 'target_polarity']
    df_cols = ['word', 'aspect', 'opinion', 'sentiment']
    labels_df = pd.DataFrame(columns=labels)
    r_i = 0

    # Create writers for 
    #       Sentence  -->        sentence.txt
    #       Aspect    -->          target.txt
    #       Opinion   -->         opinion.txt
    #       Sentiment --> target_polarity.txt
    subsets = ['train', 'val', 'test']
    writers = dict()
    for subset in subsets:
        subset_dir = os.path.join(label_dir, subset)
        if not os.path.isdir(subset_dir):
            os.makedirs(subset_dir)
        writers[subset] = dict()
        for L in labels:
            L_path = os.path.join(subset_dir, f'{L}.txt')
            writers[subset][L] = open(L_path, 'w')

    # Processing
    vocab = []
    max_length = -1
    files = glob(os.path.join(annotation_dir, '*annotation*'))
    for fn in files:
        print(f"\n\n\nProcessing {ntpath.split(fn)[-1]} ...")
        with open(fn, 'r', encoding='utf-8') as f_reader:
            annotations = f_reader.readlines()
        for annotation_unparsed in annotations:
            annotation = json.loads(annotation_unparsed)
            # print(json.dumps(annotation, indent=4, sort_keys=True))
            _, located_words_df = encode_words_location(annotation['text'])
            words_annotated = annotation['labels'] if 'labels' in annotation.keys() else []
            words_data = [a+[annotation['text'][a[0]:a[1]]] for a in words_annotated]
            try:
                words_labeled = decode_words_location(located_words_df, words_data)
            except Exception as e:
                print(f"{e}: {annotation['text']}")
                continue

            # Sentencize and Group
            sentence_ids = words_labeled.sentence_id.unique()
            for N in range(len(sentence_ids)):
                for ws in window_slide(sentence_ids, N+1):
                    words_in_sentences = words_labeled[words_labeled.sentence_id.isin(ws)]

                    # Feed label into DataFrame
                    label_data = []
                    for col in df_cols:
                        data = words_in_sentences[col].values.tolist()
                        label_data.append(' '.join(str(x) for x in data))
                    labels_df.loc[r_i] = label_data
                    r_i += 1

            # Build vocabulary
            words = words_labeled.word.values.tolist()
            vocab.extend(words)
            max_length = max(len(words), max_length)

    # labels_df.to_csv('labels.csv', index=False)

    # Make vocabulary as dict
    vocab = ['<pad>'] + list(set(vocab))
    vocab2idx = {word: w_i for w_i, word in enumerate(vocab)}
    with open(os.path.join(data_dir, f'word2id_{len(vocab)}words_maxLen{max_length}.txt'), 'w') as vocab_writer:
        vocab_writer.write(json.dumps(vocab2idx))

    # Train-Validation-Test Split
    train_test_data = train_test_split(labels_df, train_size=train_val_test_ratio[0]/sum(train_val_test_ratio), random_state=random_seed)
    val_test_data = train_test_split(train_test_data[1], test_size=train_val_test_ratio[2]/sum(train_val_test_ratio[1:3]), random_state=random_seed)
    data_subsets = [train_test_data[0]] + val_test_data
    for subset, dataset in zip(subsets, data_subsets):
        for L in labels:
            print(f"\n\nWriting to {subset}/{L}.text")
            data = dataset[L].values.tolist()
            writers[subset][L].write('\n'.join(data))

    # Close all writers
    for subset in subsets:
        for L in labels:
            writers[subset][L].close()



























