import modeling
import tokenization
import os
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras import optimizers
import keras.backend as K
from keras.callbacks import Callback
from keras_bert.loader import load_trained_model_from_checkpoint
from keras_contrib.layers import CRF
import heapq


from keras.utils import multi_gpu_model
from pprint import pprint

IS_ON_SERVER = 1

if IS_ON_SERVER == 1:
    bert_dir = '/root/xcn/bert_base/uncased_L-12_H-768_A-12/'
elif IS_ON_SERVER == 2:
    bert_dir = '/home/grayluck/xcn/bert_ch/uncased_L-12_H-768_A-12/'
else:
    bert_dir = '/Users/xcn/Desktop/workspace/bert_ch/chinese_L-12_H-768_A-12/'


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
K.set_session(sess)
# K.set_learning_phase(1)

np.random.seed(2019)
tf.set_random_seed(2019)

bert_config_file = bert_dir + 'bert_config.json'
vocab_file = bert_dir + 'vocab.txt'
init_checkpoint = bert_dir + 'bert_model.ckpt'

batch_size = 16
seq_length = 100


# test_train_size = -1
# test_predict_size = -1

#########
# training=False   可以返回编码层
# trainable=True   可以训练参数

def seq_padding(input, seq_length):
    if len(input) > seq_length:
        input = input[:seq_length]
    else:
        input += [0]*(seq_length-len(input))

    return input

def find_sublist_pos(text_list, pattern, seq_length):
    '''
    input: list str, text pattern
    output: int int, start_pos len
    '''

    pattern = tokenizer.tokenize(pattern)
    pos = -1
    offset = len(pattern)
    max_len = min(len(text_list), seq_length)
    for i in range(max_len - offset + 1):
        flag = 1
        for j in range(offset):
            if pattern[j] != text_list[i + j]:
                flag = 0
        if flag == 1:
            pos = i
            break
    return pos, offset


tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
text = 'It is an important task which can narrow down the range of candidates for an entity mention, ' \
       'and is therefore beneficial for a large number of natural language processing (NLP) tasks such ' \
       'as entity linking, relation extraction, question answering and knowledge base population.'

test_tokens = tokenizer.tokenize(text)
test_segments = [0] * len(test_tokens)
test_masked = [1] * len(test_tokens)

def mask_a_word(text_list, word):
    masked_index = text_list.index(word)
    text_list[masked_index] = '[MASK]'
    return text_list

def mask_sublist(text_list, pattern):
    start_pos, len = find_sublist_pos(text_list, pattern, seq_length)
    for i in range(len):
        text_list[start_pos+i] = '[MASK]'
    return text_list, start_pos, len


test_tokens, subtoken_start, subtoken_len = mask_sublist(test_tokens, 'extraction')
# test_tokens = mask_a_word(test_tokens, 'as')


test_token_ids = tokenizer.convert_tokens_to_ids(test_tokens)

test_token_ids = np.array([seq_padding(test_token_ids, seq_length)])
test_segments = np.array([seq_padding(test_segments, seq_length)])
test_masked = np.array([seq_padding(test_masked, seq_length)])



bert_model = load_trained_model_from_checkpoint(bert_config_file, init_checkpoint,
                                           training=True, trainable=True,
                                           seq_len=seq_length)

# output_layer = bert_model.get_layer(name='Encoder-12-FeedForward-Norm').output
# input_masked_layer = bert_model.get_layer(name='Input_Masked').input
# my_model = Model(input=bert_model.input, output=output_layer)

# 可视化
# from keras.utils import plot_model
# plot_model(bert_model, to_file='../pics/bert_model.png', show_shapes=True)
test_input = [test_token_ids, test_segments, test_masked]
test_result = bert_model.predict(test_input)[0][0]

# test_result = np.argmax(test_result,axis=-1)
# test_result = tokenizer.convert_ids_to_tokens(test_result)

print(np.shape(test_result))
print(test_tokens)

total_res = []
for i in range(subtoken_len):
    word_result_dict = {}
    word_result = list(test_result[subtoken_start+i])
    maxprob_list = heapq.nlargest(3, word_result)
    maxids_list = list(map(word_result.index, maxprob_list))
    maxword_list = tokenizer.convert_ids_to_tokens(maxids_list)


    print(maxprob_list)
    print(maxword_list)
    print()
# print(np.shape(bert_model.predict(test_input)[0][0]))

