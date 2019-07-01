import numpy as np
import os

def get_wordvec(file):
    with open(file, 'r', encoding='utf-8') as f:
        words = []
        word_to_vec_map = []
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.append(curr_word)
            word_to_vec_map.append(np.array(line[1:], dtype=np.float64))
        en_idx_dict = {}

        for i in range(len(words)):
            en_idx_dict[words[i]] = i
        idx_en_dict = {val: key for key, val in en_idx_dict.items()}
    return en_idx_dict, idx_en_dict, np.reshape(word_to_vec_map, newshape=[-1, len(word_to_vec_map[0])])


def process_MR_dataset(path_neg, path_pos):
    pass


class Config_glove:
    epoch = 20
    batch_size = 200
    learning_rate = 0.0005
    sequence_length = 5
    # number of categories
    num_class = 2
    # golve_embedding_dictionary
    en_idx_dict, idx_en_dict, vec_list = get_wordvec('../glove_wordvec/glove.6B.100d.txt')
    # size of vocabulary
    vocab_size = len(en_idx_dict)
    # size of embedding
    embedding_size = vec_list.shape[1]
    # height of filter
    filter_size = [2, 3, 4]
    # total filters -- filter_category * num of each filter
    filter_total = 18
    # number of filter
    filter_num = 6
    # filter stride in height
    filter_stride = 1
    #
    l2_reg_lambda = .1
    dropout_keep_prob = 0.5


path_neg = '../test_dataset/txt_sentoken/neg'
path_pos = '../test_dataset/txt_sentoken/pos'
process_MR_dataset(path_neg, path_pos)
