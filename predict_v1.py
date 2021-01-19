import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from models.RACL_v1 import *
from utils import *


# Define arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='racl', type=str, help='model name')
parser.add_argument('--task', default='hotel', type=str, help='task name')
parser.add_argument('--ckpt', default=69, type=int, help='checkpoint id to load weights')
parser.add_argument('--batch_size', default=16, type=int, help='number of samples per batch')
parser.add_argument('--global_dim', default=300, type=int, help='dimension of global embedding')
parser.add_argument('--domain_dim', default=100, type=int, help='dimension of domain-specific embedding')
parser.add_argument('--max_sentence_len', default=106, type=int, help='maximum number of words in sentence')
parser.add_argument('--n_interactions', default=4, type=int, help='number of RACL blocks to interact')
parser.add_argument('--n_filters', default=64, type=int, help='number of filters in convolution')
parser.add_argument('--kernel_size', default=3, type=int, help='kernel size in convolution')
parser.add_argument('--n_classes', default=3, type=int, help='number of classes')
parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
parser.add_argument('--threshold', default=0.5, type=float, help='threshold for predict')
opt = parser.parse_args()

opt.kp1, opt.kp2 = 1.0, 1.0
opt.emb_dim = opt.global_dim + opt.domain_dim
opt.finetune_embeddings = False

np.random.seed(opt.random_seed)
tf.set_random_seed(opt.random_seed)


# Define useful directories
work_dir = os.getcwd()
root_dir = os.path.dirname(work_dir)
# sourceccode_dir = os.path.dirname(root_dir)
# project_dir = os.path.dirname(sourceccode_dir)
predicts_dir = os.path.join(root_dir, 'outputs', 'predictions')

opt.data_path = os.path.join(work_dir, 'data', opt.task)
opt.vocab_path = os.path.join(opt.data_path, 'word2id.txt')
opt.global_WE_path = os.path.join(opt.data_path, 'global_embeddings.npy')
opt.domain_WE_path = os.path.join(opt.data_path, 'domain_embeddings.npy')
# opt.train_path = os.path.join(opt.data_path, 'train')
# opt.test_path = os.path.join(opt.data_path, 'test')
# opt.val_path = os.path.join(opt.data_path, 'val')
opt.ckpt_path = os.path.join(work_dir, 'checkpoints', opt.task, f"RACL.ckpt-{opt.ckpt}")


def main(_):

    # For generating the word-idx mapping and the word vectors from scratch, just run embedding.py.
    print('Reuse Word Dictionary & Embedding')
    with open(opt.vocab_path, 'r', encoding='utf-8') as f_reader:
        word2idx = json.load(f_reader)
        idx2word = {idx: word for word, idx in word2idx.items()}
    w2v = np.load(opt.global_WE_path)
    w2v_domain = np.load(opt.domain_WE_path)

    # Load architecture and weights
    model = RACL(opt, w2v, w2v_domain, word2idx, 'predict')
    weights_loaded = model.load_weights(opt.ckpt_path)
    if not weights_loaded:
        print('Fail to load model weights')
        quit()

    # Samples for prediction
    documents = [
        # 'dessert was also to die for', 
        # 'sushi so fresh that it crunches in your mouth',
        # 'in fact , this was not a nicoise salad and was barely eatable',
        # "the two waitress 's looked like they had been sucking on lemons",
        # "the absence of halal food - not even for room service",
        # "Have to travel out in order to get food",
        # "Smell of the pillows... smelt like someone odour"
        " Very noisy outside the room, found a cockroaches in bathroom, the condition did not works whole nights, very hot can't sleep",
        "I had to stay here due to holiday inn transferring me here because they were closed for renovations. First I am pist because this hotel stinks of weed, my room was not very clean and due to Covid you would think the room would be super clean but nope wrappers all over the place towels had stains, to top it off I even found bugs in my room. I am disgusted. The service is horrible. “There was never a manager on duty” I even reached out to them in email and still no reply from them so they clearly don’t care. Avoid this hotel there are so many other options by the airport that this one poor excuse for cleanliness and bugs they do not deserve a dime. They don’t fix their problems and a manager is never reachable",
        "First impression is the hotel seem to be in need of an upgrade. The grounds did not feel welcoming on the exterior. The interior had carpet coming up in the hallway, I was on the third floor. It had a bad smell that hits you in the face as soon as you get off the elevator. The rooms was decent with a nice size television, desk and a refrigerator but lacked cleanliness. We couldn't shower because the tubes were GROSS. It looked as if it hadn't been properly cleaned for months! You can see the filth buildup YUCK! This is very concerning considering the month I traveled was during the covid-19 pandemic. If this hotel is not properly cleaning guest rooms than are they really practicing safe measures during a global coronavirus pandemic?",
        "Small rooms, restaurant offers the best of microwaved food and wifi is poor. Staff set engaged, but this establishment needs investment and attention to the the customer experience. Plenty of examples where the site could use a goos cleaning - including the restaurant.",
        "I had a horrible check-in experience at this crown plaza. The manager at night shift was exceptionally rude. Just because it was night and I was tired, I stayed there. I checked out next day and went to The Renaissance across the street.",
        "DIRTY FILTHY DISGUSTING!!! Hair and mold in the bathroom, DIRTY carpeting, smells of cigarette smoke and my daughter woke up with bug bites all over her legs!!! Front desk was an absolute joke! Unprofessional rude and lazy!! Travelers BEWARE!!",
        "Called to say my flight is cancelled because of weather ,can you change to next day or refund.before I could complete the sentence they cancelled my reservation and hung up.i know the hotel room was given to somebody else.i cannot believe the service was from very reputable company like yours",
        "The value for the room and the service was very good but the Furnishings in the room is very outdated and more out. The carpet has been replaced and the linen and the bathtub was spotless. Restaurant bar",
        "The Crowne Plaza is located near the newark airport. The hotel offers a transfer ( i got it on my way back). The rooms are small but the bed is very comfortable. Bathroom regular. Also offers a transfer to the outlet nearby but only in 2 specific times a day.",
        "We stayed one night (thankfully) as there was a lot of noise from airplanes taking off and landing and from traffic on the road nearby. The room was very nice with comfortable bed. The shower was over the bath",
        "I visited this hotel with 6 family members in jan 2020. we reached jetlagged early in the morning to be greeted by an extremely rude lady whose name started with Q. I saw her even mocking a few clients. Rooms were clean. Sleep quality was nice Not many eating options around hotel for breakfast, except the hotel itself. In evening one can walk out towards quay and be delighted with so many restaurants. over all a an average hotel BUT the RUDEST STAFF i have ever seen. STAY AWAY IF YOU ANYOTHER OPTION.",
        "Hotel was very crowded and so called club lounge was so crowded that we couldn't use 20 minute wait for breakfast in main restaurant Hotel room small and basic - not luxury Pool good and hotel location excellent",
        "The hotel is actually Robertson Quay not Clarke Quay as the name claims. I had booked a room with a king size bed but they could only give me twin beds on the first night so I had to move rooms on the second day. All of the rooms I saw were tired with very bland decor and badly in need of a refresh. I also experienced a lot of noise from neighbouring rooms",
        "I do no understand why you are charging me USD 100 (66% of original room charge) because I have Netherlands nationality but booked my room stating my residential address in Thailand, where I have lived for the last 13 years",
        "Check in was appalling ! Checked into a deluxe room but was given two single beds!! Went downstairs to speak to reception and they told me only room they have is a smoking room which was not practical!!! Then had to sleep there and next day await a room change!!! Which was chased by us as no one remembered the next day!!",
        "I would not recommend this hotel, it is seriously understaffed the restaurant is small for the size of the hotel which results in the tables being too close together. The restaurant staff tried their best but there just weren't enough of them",
        "nice bar and front desk staff members happy faces they made me feel like a vip. update! hotel is dark and old. bathroom was tiny, dark and poor design. elevator was slow. hotel facilities and staff were excellent",
    ]

    # Split document into sentences
    sentences, sent2doc = split_documents(documents)
    opt.batch_size = len(sentences)

    # Predict
    start_time = time.time()
    sentences_input, sentences_mask, position_matrices, original_sentences = tokenize(sentences, word2idx, opt.max_sentence_len, '<oov>')
    sentences_input = np.reshape(sentences_input, (opt.batch_size, opt.max_sentence_len))
    aspect_probs, opinion_probs, sentiment_probs = model.predict(
        sentences_input, 
        np.reshape(sentences_mask, (opt.batch_size, opt.max_sentence_len)), 
        np.reshape(position_matrices, (opt.batch_size, opt.max_sentence_len, opt.max_sentence_len))
    )
    end_time = time.time()
    time_running = end_time - start_time
    run_time = 'Predicting in {:.0f}m {:.0f}s'.format(time_running//60, time_running%60)
    print(run_time)

    # Feed results into DataFrame
    results_df = decode_results(original_sentences, sent2doc, sentences_input, 
                                aspect_probs, opinion_probs, sentiment_probs)

    # Write logs
    output_file = os.path.join(predicts_dir, f'case_study_{opt.task}')
    # write_log(results_df, f'case_study_{opt.task}.log')
    # write_html(results_df, f'case_study_{opt.task}.html')
    doc_results = format_results(results_df)
    with open(output_file+'.json', 'w') as f_writer:
        json.dump(doc_results, f_writer, indent=4)
    dict2html(doc_results, output_file+'.html')


if __name__ == '__main__':
    tf.app.run()
















