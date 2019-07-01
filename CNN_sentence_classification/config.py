import numpy as np
import os
import nltk
import datetime
import math

def get_wordvec(file):
    with open(file, 'r', encoding='utf-8') as f:
        words = []
        word_to_vec_map = []
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.append(curr_word)
            word_to_vec_map.append(np.array(line[1:], dtype=np.float64))
        words.append("PADDING")
        word_to_vec_map.append(np.zeros([np.array(word_to_vec_map).shape[1]], dtype=np.float64))
        en_idx_dict = {}

        for i in range(len(words)):
            en_idx_dict[words[i]] = i
        idx_en_dict = {val: key for key, val in en_idx_dict.items()}
    return en_idx_dict, idx_en_dict, np.reshape(word_to_vec_map, newshape=[-1, len(word_to_vec_map[0])])


def process_MR_dataset(path_neg, path_pos):
    config = Config_glove()
    sen = []
    y = []
    x = []
    x_output = []
    label = [[0, 1], [1, 0]]
    sen_num = 0
    for path in [path_neg, path_pos]:
        path_num = 0 if path == path_neg else 1
        files = os.listdir(path)
        for file in files:
            with open(path + '/' + file, "r") as f:
                data = f.readlines()
                for row in data:
                    sen_num += 1
                    sen.append(row)
                    y.append(label[path_num])
    for s in sen:
        words = nltk.word_tokenize(s)
        if len(words) > config.sequence_length:
            words = words[:config.sequence_length]
        else:
            for i in range(config.sequence_length-len(words)):
                words.append("PADDING")
        x.append(words)
    en_idx_dict, _, _ = get_wordvec('../glove_wordvec/glove.6B.100d.txt')
    for word_list in x:
        idx_list = [en2idx(word, en_idx_dict) for word in word_list]
        x_output.append(idx_list)
    return x_output, y


def en2idx(word, en_idx_dict):
    if word in en_idx_dict.keys():
        return en_idx_dict[word]
    else:
        return en_idx_dict['unknown']


def write(list, filename):
    with open('../test_dataset/' + filename, 'w') as f:
        f.write(str(list))


class Config_glove:
    epoch = 20
    batch_size = 200
    learning_rate = 0.0005
    # the length of sentence
    sequence_length = 30
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


if __name__ == '__main__':
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    path_neg = '../test_dataset/txt_sentoken/neg'
    path_pos = '../test_dataset/txt_sentoken/pos'
    x, y = process_MR_dataset(path_neg, path_pos)
    write(x, 'x.txt')
    write(y, 'y.txt')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))




# nltk source not found
# import ssl
# import nltk
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     # Legacy Python that doesn't verify HTTPS certificates by default
#     pass
# else:
#     # Handle target environment that doesn't support HTTPS verification
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('punkt')