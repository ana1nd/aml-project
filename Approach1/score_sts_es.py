#!/usr/bin/python 
from __future__ import division
import sys
import unicodedata
import re
import os
import torch
import nltk
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
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

if __name__ == '__main__':
    
    arguments = sys.argv
    script = arguments[0]
    model_name = arguments[1]

    data_dir = 'test_data_es'
    input_files = ['sts15.input.newswire.txt','sts15.input.wikipedia.txt','sts17.input.track3.es-es.txt']
    gs_files = ['sts15.gs.newswire.txt','sts15.gs.wikipedia.txt','sts17.gs.track3.es-es.txt']
    for i in range(0,len(input_files),1):
        file = input_files[i]
        gs_file = gs_files[i]
        name = file.split('.')
        output = name[0] + '.output.' + ".".join(name[2:])
        #f = open(output,'w')
        
        file_path = os.path.join(data_dir,file)
        lines = open(file_path).read().strip().split('\n')
        
        gs_file_path = os.path.join(data_dir,gs_file)
        gs_lines = open(gs_file_path).read().strip().split('\n')
        gs_lines = [float(i) for i in gs_lines]
        first, second = list(), list()
        for i in range(0,len(lines),1):
            a,b = lines[i].split('\t')
            a_clean,b_clean = normalizeString(a),normalizeString(b)            
            first.append(a_clean)
            second.append(b_clean)
        
    
        print("File processed")
        print len(first),len(second)
        print first[0]
        print "************"
        print second[0]
        model_dir = "encoderdir"
        encoder_path = os.path.join(model_dir,model_name)
    
        glove = "es.vec"
        glove_path = os.path.join(os.getcwd(),glove)
    
    
        if torch.cuda.is_available():
            infersent = torch.load(encoder_path) #GPU
        else:    
            infersent = torch.load(encoder_path, map_location=lambda storage, loc: storage)
        
    
        infersent.set_glove_path(glove_path)    
        infersent.build_vocab(first, tokenize=True)
        infersent.build_vocab(second,tokenize=True)
    
        firstv = infersent.encode(first, tokenize=True)
        secondv = infersent.encode(second,tokenize=True)    

        print firstv.shape,secondv.shape
        cos = list()
        for i in range(0,len(firstv),1):
            #cos = cosine_similarity([firstv[i]],[secondv[j]])
            cos.append(distance.cosine(firstv[i],secondv[i]))
            #line = str(cos) + "\n"
            #f.write(line)
            
        mn = min(gs_lines)
        mx = max(gs_lines)
        
        lb = np.floor(mn)
        ub = np.ceil(mx)
        
        print min(gs_lines),max(gs_lines),lb,ub,min(cos),max(cos)
        gs_norm = normalize(gs_lines,lb,ub)
        print min(gs_norm),max(gs_norm),lb,ub,min(cos),max(cos)
        coeff , p = scipy.stats.pearsonr(gs_norm,cos)
        print scipy.stats.pearsonr(gs_norm,cos),scipy.stats.spearmanr(gs_norm,cos)

        print "=========================================================="
    
    
    
    
    
    
    
    

    