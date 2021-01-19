import os
import time
import argparse
import gc
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product

from models.RACL_v1 import *
from utils import *


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.logging.set_verbosity(tf.logging.ERROR)


# Define useful directories
working_dir = os.getcwd()
root_dir = os.path.dirname(working_dir)
ref_dir = os.path.dirname(root_dir)
project_dir = os.path.dirname(ref_dir)


# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='racl', type=str, help='model name')
parser.add_argument('--task', default='hotel', type=str, help='hotel domain')
parser.add_argument('--n_epochs', default=11, type=int, help='number of epochs for training a whole dataset')
parser.add_argument('--batch_size', default=16, type=int, help='number of samples per batch')
parser.add_argument('--global_dim', default=300, type=int, help='dimension of global embedding')
parser.add_argument('--domain_dim', default=100, type=int, help='dimension of domain-specific embedding')
parser.add_argument('--max_sentence_len', default=106, type=int, help='maximum number of words in sentence')
parser.add_argument('--n_classes', default=3, type=int, help='number of classes')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer for model | default: SGD (Stochastic Gradient Descent)')
parser.add_argument('--aspect_weight', default=1.1, type=float, help='weight of aspect loss')
parser.add_argument('--opinion_weight', default=1.3, type=float, help='weight of opinion loss')
parser.add_argument('--sentiment_weight', default=1.0, type=float, help='weight of sentiment loss')
parser.add_argument('--regularization_weight', default=1e-4, type=float, help='weight of regularization loss')
parser.add_argument('--load_pretrained', default=False, type=bool, help='whether to load an existing checkpoint')
parser.add_argument('--finetune_embeddings', default=False, type=bool, help='whether to train both word embeddings')
parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
opt = parser.parse_args()


opt.kp1, opt.kp2 = 1.0, 1.0
opt.n_epochs = 10
opt.warmup_epochs = opt.n_epochs
opt.emb_dim = opt.global_dim + opt.domain_dim
opt.load_pretrained = False


LRs = [1e-5, 5e-5, 1e-4] # learning rate
FILTERs = [64, 86] # number of filters for convolution
KERNELs = [3, 5] # kernel size for convolution
STACKs = [3, 4] # number of RACL stacks


# Define useful directories
work_dir = os.getcwd()
root_dir = os.path.dirname(work_dir)
# sourceccode_dir = os.path.dirname(root_dir)
# project_dir = os.path.dirname(sourceccode_dir)

opt.data_path = os.path.join(work_dir, 'data', opt.task)
opt.vocab_path = os.path.join(opt.data_path, 'word2id.txt')
opt.global_WE_path = os.path.join(opt.data_path, 'global_embeddings.npy')
opt.domain_WE_path = os.path.join(opt.data_path, 'domain_embeddings.npy')
opt.train_path = os.path.join(opt.data_path, 'train')
opt.test_path = os.path.join(opt.data_path, 'test')
opt.val_path = os.path.join(opt.data_path, 'val')


def main(_):
    # For generating the word-idx mapping and the word vectors from scratch, just run embedding.py.
    print('Reuse Word Dictionary & Embedding')
    with open(opt.vocab_path, 'r', encoding='utf-8') as f_reader:
        word_dict = json.load(f_reader)
    w2v = np.load(opt.global_WE_path)
    w2v_domain = np.load(opt.domain_WE_path)

    # DataFrame to contain results for comparison
    result_df = pd.DataFrame(
        columns=['lr', 'n_filters', 'kernel', 'n_stacks', 'f1_a', 'f1_o', 'f1_s', 'f1_absa', 'time [sec/ep]'])
    
    # Hyperparameter Tuning
    for hp_i, (lr, n_filters, kernel, n_stacks) in enumerate(product(LRs, FILTERs, KERNELs, STACKs)):

        print(f'\n\n\nExperiment with lr={lr}, n_filters={n_filters}, kernel_size={kernel}, n_stacks={n_stacks} ...')

        # Assign hyperparams
        opt.learning_rate = lr
        opt.n_filters = n_filters
        opt.kernel_size = kernel
        opt.n_interactions = n_stacks

        # Run experiment
        model = RACL(opt, w2v, w2v_domain, word_dict, 'train')
        start_time = time.time()
        f1_e, f1_a, f1_s, f1_absa = model.train(is_tuning=True)
        end_time = time.time()
        total_time = end_time - start_time
        time_per_epoch = total_time / opt.n_epochs
        result_df.loc[hp_i] = [lr, n_filters, kernel, n_stacks, f1_a, f1_e, f1_s, f1_absa, time_per_epoch]
        print(f'\t... in {total_time:.2f} seconds')

        # Release model
        tf.keras.backend.clear_session()
        del model
        gc.collect()

    fn = os.path.join(work_dir, 'hyperparams_logs', opt.task, 'hyperparams.csv')
    result_df.to_csv(fn, index=False)


if __name__ == '__main__':
    tf.app.run()










