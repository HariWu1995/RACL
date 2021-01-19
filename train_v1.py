import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
import argparse

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from models.RACL_v1 import *
from utils import *


# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='racl', type=str, help='model name')
parser.add_argument('--task', default='hotel', type=str, help='task name')
parser.add_argument('--n_epochs', default=69, type=int, help='number of epochs for training a whole dataset')
parser.add_argument('--warmup_epochs', default=1, type=int, help='number of epochs for warmup (without evaluation)')
parser.add_argument('--batch_size', default=16, type=int, help='number of samples per batch')
parser.add_argument('--learning_rate', default=0.00069, type=float, help='learning rate')
parser.add_argument('--global_dim', default=300, type=int, help='dimension of global embedding')
parser.add_argument('--domain_dim', default=100, type=int, help='dimension of domain-specific embedding')
parser.add_argument('--max_sentence_len', default=106, type=int, help='maximum number of words in sentence')
parser.add_argument('--n_interactions', default=4, type=int, help='number of RACL blocks to interact')
parser.add_argument('--kp1', default=0.5, type=float, help='keep prob for inputs')
parser.add_argument('--kp2', default=0.5, type=float, help='keep prob for tasks')
parser.add_argument('--n_filters', default=64, type=int, help='number of filters in convolution')
parser.add_argument('--kernel_size', default=3, type=int, help='kernel size in convolution')
# parser.add_argument('--n_classes', default=3, type=int, help='number of classes')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer for model | default: SGD (Stochastic Gradient Descent)')
parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
parser.add_argument('--aspect_weight', default=1.1, type=float, help='weight of aspect loss')
parser.add_argument('--opinion_weight', default=1.3, type=float, help='weight of opinion loss')
parser.add_argument('--sentiment_weight', default=1.0, type=float, help='weight of sentiment loss')
parser.add_argument('--regularization_weight', default=1e-3, type=float, help='weight of regularization loss')
parser.add_argument('--finetune_embeddings', default=False, type=bool, help='whether to train both word embeddings')
parser.add_argument('--load_pretrained', default=False, type=bool, help='whether to load an existing checkpoint')
opt = parser.parse_args()

opt.n_classes = 3
if opt.warmup_epochs < 0:
    opt.warmup_epochs = opt.n_epochs // 11
opt.emb_dim = opt.global_dim + opt.domain_dim

np.random.seed(opt.random_seed)
tf.set_random_seed(opt.random_seed)


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
        # word_dict = eval(f_reader.read())
    w2v = np.load(opt.global_WE_path)
    w2v_domain = np.load(opt.domain_WE_path)

    # Train
    start_time = time.time()
    model = RACL(opt, w2v, w2v_domain, word_dict, 'train')
    model.train()
    end_time = time.time()
    time_running = end_time - start_time
    run_hours = time_running // 3600
    run_minutes = (time_running-run_hours*60) // 60
    run_seconds = time_running - run_hours*3600 - run_minutes*60
    run_time = '\n\n\nTraining in {}h {}m {}s'.format(int(run_hours), int(run_minutes), int(run_seconds))
    print(run_time)
    
    # Save global and domain word embeddings
    # np.save(opt.global_WE_path.replace('.npy', '_finetuned.npy'), model.w2v)
    # np.save(opt.domain_WE_path.replace('.npy', '_finetuned.npy'), model.w2v_domain)


if __name__ == '__main__':
    tf.app.run()
















