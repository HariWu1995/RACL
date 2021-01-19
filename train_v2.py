import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
import argparse

import random
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3) # 0: debug, 1: info, 2: warning, 3: error

from models.RACL_v2 import *
from utils import *


# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='racl', type=str, help='model name')
parser.add_argument('--task', default='hotel', type=str, help='task name')
parser.add_argument('--n_epochs', default=69, type=int, help='number of epochs for training a whole dataset')
parser.add_argument('--batch_size', default=16, type=int, help='number of samples per batch')
parser.add_argument('--lr', default=0.0069, type=float, help='learning rate')
parser.add_argument('--global_dim', default=300, type=int, help='dimension of global embedding')
parser.add_argument('--domain_dim', default=100, type=int, help='dimension of domain-specific embedding')
parser.add_argument('--max_sentence_len', default=106, type=int, help='maximum number of words in sentence')
parser.add_argument('--n_interactions', default=4, type=int, help='number of RACL blocks to interact')
parser.add_argument('--keep_prob_1', default=1., type=float, help='keep prob for inputs')
parser.add_argument('--keep_prob_2', default=1., type=float, help='keep prob for tasks')
parser.add_argument('--n_filters', default=64, type=int, help='number of filters in convolution')
parser.add_argument('--kernel_size', default=3, type=int, help='kernel size in convolution')
# parser.add_argument('--n_classes', default=3, type=int, help='number of classes')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer for model | default: SGD (Stochastic Gradient Descent)')
parser.add_argument('--random_type', default='normal', type=str, help='random type: uniform or normal (default)')
parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
parser.add_argument('--aspect_weight', default=1., type=float, help='weight of aspect loss')
parser.add_argument('--opinion_weight', default=1., type=float, help='weight of opinion loss')
parser.add_argument('--sentiment_weight', default=1.3, type=float, help='weight of sentiment loss')
parser.add_argument('--regularization_weight', default=1e-4, type=float, help='weight of regularization loss')
parser.add_argument('--label_smoothing', default=0., type=float, help='label smoothing for regularization')
parser.add_argument('--finetune_embeddings', default=False, type=bool, help='whether to train both word embeddings')
parser.add_argument('--feed_text_embeddings', default=False, type=bool, help='whether to feed word embeddings as inputs')
parser.add_argument('--load_pretrained', default=False, type=bool, help='whether to load an existing checkpoint')
parser.add_argument('--include_opinion', default=True, type=bool, help='whether to use opinion for model')
opt = parser.parse_args()

opt.n_classes = 3
opt.emb_dim = opt.global_dim + opt.domain_dim
# opt.n_filters = opt.emb_dim / 2
opt.is_training = True
opt.class_weights = [.2, .5, .3] # Outside - Beginning - Inside

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
tf.random.set_seed(opt.random_seed)


# Define useful directories
work_dir = os.getcwd()
root_dir = os.path.dirname(work_dir)
sourceccode_dir = os.path.dirname(root_dir)
project_dir = os.path.dirname(sourceccode_dir)

opt.data_path = os.path.join(work_dir, 'data', opt.task)
opt.vocab_path = os.path.join(opt.data_path, 'word2id.txt')
opt.global_WE_path = os.path.join(opt.data_path, 'global_embeddings.npy')
opt.domain_WE_path = os.path.join(opt.data_path, 'domain_embeddings.npy')
opt.train_path = os.path.join(opt.data_path, 'train')
opt.test_path = os.path.join(opt.data_path, 'test')
opt.val_path = os.path.join(opt.data_path, 'val')



def main():

    # For generating the word-idx mapping and the word vectors from scratch, just run embedding.py.
    if opt.feed_text_embeddings:
        model_path = os.path.join(project_dir, '2. Data Acquisition', 'FastText_pretrained', 'cc.en.128.bin')
        opt.word_dict = load_fasttext_model(model_path)
        opt.emb_dim = 128
    else:
        with open(opt.vocab_path, 'r', encoding='utf-8') as f_reader:
            opt.word_dict = json.load(f_reader)
            opt.vocab_size = len(opt.word_dict)
        opt.w2v = np.load(opt.global_WE_path)
        opt.w2v_domain = np.load(opt.domain_WE_path)

    # Train
    start_time = time.time()
    model = RACL(opt)
    model.train()
    end_time = time.time()
    time_running = end_time - start_time
    run_hours = time_running // 3600
    run_minutes = (time_running-run_hours*60) // 60
    run_seconds = time_running - run_hours*3600 - run_minutes*60
    run_time = '\n\n\nTraining in {}h {}m {}s'.format(int(run_hours), int(run_minutes), int(run_seconds))
    print(run_time)
    
    # Save global and domain word embeddings
    if opt.finetune_embeddings:
        np.save(opt.global_WE_path.replace('.npy', '_finetuned.npy'), model.w2v)
        np.save(opt.domain_WE_path.replace('.npy', '_finetuned.npy'), model.w2v_domain)


if __name__ == '__main__':
    main()
















