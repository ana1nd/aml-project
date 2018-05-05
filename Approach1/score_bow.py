#!/usr/bin/python 
from __future__ import division
import sys
import unicodedata
import re
import os
import torch
import nltk
import numpy as np
import pickle
import scipy.spatial.distance as distance
import scipy


def unicodeToAscii(s):

    temp = []
    for s_ in s.split():
        temp.append(unicodedata.normalize('NFKD',unicode(s_,'utf8')).encode('ascii','ignore'))
    s = ' '.join(temp)
    return s

def normalizeString(s):
    
    s = unicodeToAscii(s.lower().strip())
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)    
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    
    stopwords = ['apos','quot']
    querywords = s.split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    s = ' '.join(resultwords)    
    
    return s

def normalize(vec,lb,ub):
    norm_vec = []
    for i in range(0,len(vec),1):
        a = (vec[i] - lb)/(ub-lb)
        norm_vec.append(a)
    return norm_vec

def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return pickle.load(f)    

def get_glove(glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    s = set()
    #file_name = 'word_vec_es'
    file_name = 'dictionary' +'.pkl'
    set_name = 'set.pkl'
    print file_name,os.path.isfile(file_name)
    if os.path.isfile(file_name):
        print " File already present"
        word_vec = load_obj(file_name)
        s = load_obj(set_name)
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

            if word not in s:
                s.add(word)
            #vec = vec.split(' ')
            #print word
            #print vec
            vec = [float(i) for i in vec]
            #print lineN,word,len(vec)
            assert len(vec) ==  300
            vec = np.array(vec)
            word_vec[word] = vec
            lineN += 1
        save_obj(word_vec,file_name)
        save_obj(s,set_name)
    return word_vec, s   

if __name__ == '__main__':

    data_dir = 'test_data_es'
    input_files = ['sts15.input.newswire.txt','sts15.input.wikipedia.txt','sts17.input.track3.es-es.txt']
    gs_files = ['sts15.gs.newswire.txt','sts15.gs.wikipedia.txt','sts17.gs.track3.es-es.txt']
    for z in range(0,len(input_files),1):
        file = input_files[z]
        gs_file = gs_files[z]
        print file,gs_file
        name = file.split('.')
        output = name[0] + '.output.' + ".".join(name[2:])
        #f = open(output,'w')
        
        file_path = os.path.join(data_dir,file)
        lines = open(file_path).read().strip().split('\n')
        
        gs_file_path = os.path.join(data_dir,gs_file)
        gs_lines = open(gs_file_path).read().strip().split('\n')
        gs = [float(i) for i in gs_lines]
        mn, mx = min(gs), max(gs)
        lb, ub = np.floor(mn), np.ceil(mx)
        gs_norm = normalize(gs,lb,ub)

        
        glove = "es.vec"
        glove_path = os.path.join(os.getcwd(),glove)
        word_vec, s = get_glove(glove_path)        
        
        first, second, cos = list(), list(), list()
        for i in range(0,len(lines),1):
            a,b = lines[i].split('\t')
            a_clean,b_clean = normalizeString(a),normalizeString(b)
            a_ls , b_ls = a_clean.split(' '), b_clean.split(' ')
            
            counta = 0
            a_avg = np.zeros(300,float)
            for word in a_ls :
                if word in s:
                    a_avg += word_vec[word]
                    counta += 1

            if counta > 0:
                a_avg = a_avg/counta
            
            countb = 0
            b_avg = np.zeros(300,float)
            for word in b_ls:
                if word in s:
                    b_avg += word_vec[word]
                    countb += 1
            
            if countb > 0:
                b_avg = b_avg/countb 
            #print i,counta,len(a_ls),countb,len(b_ls)
            cos.append(distance.cosine(a_avg,b_avg))
            #cos.append(1)
            
        print min(gs_norm),max(gs_norm),lb,ub,min(cos),max(cos)
        coeff , p = scipy.stats.pearsonr(gs_norm,cos)
        #print coeff,p
        print scipy.stats.pearsonr(gs_norm,cos),scipy.stats.spearmanr(gs_norm,cos)
        print "=========================================================="            