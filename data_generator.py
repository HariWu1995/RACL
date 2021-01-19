import os
from glob import glob

import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence

from utils import *


class DataGenerator(Sequence):

    def __init__(self, data_root, opt, is_shuffle: bool=True, load_all: bool=False):
        
        self.opt = opt
        self.is_shuffle = is_shuffle
        
        self.data = read_data(data_root, opt.word_dict, opt.max_sentence_len, not opt.is_training, feed_embedding=self.opt.feed_text_embeddings)
        self.n_samples = len(self.data[0])
        if load_all:
            self.batch_size = self.n_samples
            self.n_batches = 1
        else:
            self.batch_size = opt.batch_size
            self.n_batches = int(self.n_samples/self.batch_size) + (1 if self.n_samples % self.batch_size else 0)

        self.indices = np.arange(self.n_samples)
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return self.n_batches

    def __getitem__(self, index):
        """ Generate one batch of data """

        # Generate indexes of the batch
        start_index = self.batch_size * index
        end_index = self.batch_size * (index+1)
        indices = self.indices[start_index:end_index]

        # Generate data
        batch_input = self.data[0][indices]
        batch_labels = [self.data[d_i][indices] for d_i in [1,2,3]]
        if self.opt.label_smoothing > 0.:
            for d_i in range(len(batch_labels)):
                batch_labels[d_i] = smooth_labels(batch_labels[d_i], self.opt.label_smoothing)
        batch_masks = [self.data[d_i][indices] for d_i in [4,5,6]]
        return [batch_input]+batch_masks, batch_labels # for Keras.Model.fit_generator

    def on_epoch_end(self):
        """ Update indices after each epoch """
        if self.is_shuffle:
            np.random.shuffle(self.indices)