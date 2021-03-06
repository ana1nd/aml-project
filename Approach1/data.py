# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import numpy as np
import torch
import pickle

def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
    #print batch
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]
            #print i,j,batch[i][j]
            

    return torch.from_numpy(embed).float(), lengths


def get_word_dict(sentences):
    # create vocab of words
    word_dict = {}
    for sent in sentences:
        for word in sent.split():
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict


def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)


def get_glove(word_dict, glove_path,datatype):
    # create word_vec with glove vectors
    word_vec = {}
    
    #file_name = 'word_vec_es'
    file_name = 'word_vec_es_' + datatype + '.pkl'
    print file_name,os.path.isfile(file_name)
    if os.path.isfile(file_name):
        print " File already present"
        word_vec = load_obj(file_name)
    else:
        word_vec['<s>'] = np.random.normal(0,1,300)
        word_vec['</s>'] = np.random.normal(0,1,300)
        word_vec['<p>'] = np.random.normal(0,1,300)
        print " GloVe file not present: make & save"
        lineN = 0
        lines = open(glove_path).read().strip().split('\n')
        for j in range(0,len(lines)-1,1):
            line = lines[j]
            temp = line.split(' ')
            temp = temp[:-1]
            '''print line
            print "*********"
            print temp'''
            word,vec = temp[0],temp[1:]
            if lineN % 100000 == 0:
                print lineN,word
            
            #vec = vec.split(' ')
            #print word
            #print vec
            vec = [float(i) for i in vec]
            #print lineN,word,len(vec)
            assert len(vec) ==  300
            vec = np.array(vec)
            if word in word_dict:
                word_vec[word] = vec
            #print lineN,word
            lineN += 1
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))
        save_obj(word_vec,file_name)
    return word_vec
        
'''        with open(glove_path) as f:
            lineN = 0
            for line in f:
                #word, vec = line.split(' ', 1)
                word, vec = line.split(' ',1)
                if lineN % 100000 == 0 :
                    print lineN,word
                lineN += 1
                vec = vec.split(' ')
                vec = [float(i) for i in vec]
                vec = np.array(vec)
                if word in word_dict:
                    word_vec[word] = vec #np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with glove vectors'.format(
                    len(word_vec), len(word_dict)))'''
    


def build_vocab(sentences, glove_path,datatype):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove(word_dict, glove_path,datatype)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec


def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}

    dico_label = {'entailment': 0,  'neutral': 1, 'contradiction': 2}
    #dico_label = {'entailment': 0,  'non-entailment':1}

    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path,
                                                 'labels.' + data_type)

        s1[data_type]['sent'] = [line.rstrip() for line in
                                 open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in
                                 open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')]
                for line in open(target[data_type]['path'], 'r')])

        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == \
            len(target[data_type]['data'])

        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                data_type.upper(), len(s1[data_type]['sent']), data_type))

    train = {'s1': s1['train']['sent'], 's2': s2['train']['sent'],
             'label': target['train']['data']}
    dev = {'s1': s1['dev']['sent'], 's2': s2['dev']['sent'],
           'label': target['dev']['data']}
    test = {'s1': s1['test']['sent'], 's2': s2['test']['sent'],
            'label': target['test']['data']}
    return train, dev, test
