from __future__ import division
import os
import sys
import time
import argparse
import time

import numpy as np
import torch
import torch as th
from torch.autograd import Variable
import torch.nn as nn
import pickle

from data import *
from mutils import get_optimizer
from models import NLINet

#import scipy.spatial.distance as distance
import scipy
from scipy import stats, optimize, interpolate

start_time = time.time()
GLOVE_PATH = "es.vec"


parser = argparse.ArgumentParser(description='NLI training')

# paths
parser.add_argument("--nlipath", type=str, default='dataset/MultiNLI_original/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--encoderdir",type=str,default='encoderdir/',help="Directory to save encoder")
parser.add_argument("--outputmodelname", type=str, default='model.pickle')
parser.add_argument("--encodermodelname",type=str,default='encoder.pickle',help="Directory to save encoder")
parser.add_argument("--filename",type=str,default='encoder.pickle',help="Directory to save encoder")
parser.add_argument("--logdir",type=str,default='logs/',help="Directory to save encoder")

#for WMT
parser.add_argument("--inputfile",type=str,default='../data/wmt_data/references/newstest2014-ref.cs-en',help="Directory to save encoder")
parser.add_argument("--gsfile",type=str,default='../data/wmt_data/system-outputs/newstest2014/cs-en/newstest2014.cu-moses.3383.cs-en',help="Directory to save encoder")
parser.add_argument("--flag",type=str,default='../data/wmt_data/system-outputs/newstest2014/cs-en/newstest2014.cu-moses.3383.cs-en',help="Directory to save encoder")



# training
parser.add_argument("--n_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dpout_model", type=float, default=0., help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0., help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=0, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=5, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")
parser.add_argument("--do_training", type=bool, default=True, help="flag for training")

# model
parser.add_argument("--encoder_type", type=str, default='BLSTMEncoder', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=100, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=512, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=2, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu_id
parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")


params, _ = parser.parse_known_args()


# set gpu device
print ('Available devices ', th.cuda.device_count())
print ('Current cuda device ', th.cuda.current_device())
th.cuda.set_device(1)
print ('Current cuda device ', th.cuda.current_device())
print (' Yes/No ', th.cuda.is_available(),4/3)
#torch.cuda.set_device(0)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)


"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)


"""
DATA
"""
def get_data_sts(path):
	s1 = {}
	s2 = {}
	s1['test'] = dict()
	s2['test'] = dict()
	lines = open(path).read().strip().split('\n')
	# first = []
	# second = []
	# for line in lines:
	# 	a,b = line.split('\t')
	# 	first.append(a)
	# 	second.append(b)
	s1['test']['sent'] = [line.rstrip().split('\t')[0] for line in open(path, 'r')]
	s2['test']['sent'] = [line.rstrip().split('\t')[1] for line in open(path, 'r')]
	test = {'s1': s1['test']['sent'], 's2': s2['test']['sent']}
	first, second = s1['test']['sent'], s2['test']['sent']
	print s1['test']['sent'][0]
	print s2['test']['sent'][0]
	return test,first,second

def build_vocab_sts(sentences, glove_path,name):
    word_dict = get_word_dict(sentences)
    word_vec = get_glove_sts(word_dict, glove_path,name)
    print('Vocab size : {0}'.format(len(word_vec)))
    return word_vec 

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

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

def get_glove_sts(word_dict, glove_path,name):
    # create word_vec with glove vectors
    word_vec = {}
    
    #file_name = 'word_vec_es'
    file_name = name + '.pkl'
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

test,premise,hypothesis = get_data_sts(params.inputfile)
word_vec = build_vocab_sts(test['s1'] + test['s2'], GLOVE_PATH,params.flag)

for split in ['s1', 's2']:
    for data_type in ['test']:
        eval(data_type)[split] = np.array([['<s>'] +
            [word for word in sent.split() if word in word_vec] +
            ['</s>'] for sent in eval(data_type)[split]])

params.word_emb_dim = 300


"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  False                  ,

}

# model
encoder_types = ['BLSTMEncoder', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder','GRUEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)

nli_net = NLINet(config_nli_model)
#print(nli_net)

# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
#nli_net.cuda()
loss_fn.cuda()

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None

def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    #target = valid['label'] if eval_type == 'valid' else test['label']

    entail_forward = []
    entail_backward = []
    print "s1 = ",len(s1)
    for i in range(0, len(s1), params.batch_size):
        # prepare batch
        s1_batch, s1_len = get_batch(s1[i:i + params.batch_size], word_vec)
        s2_batch, s2_len = get_batch(s2[i:i + params.batch_size], word_vec)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        #tgt_batch = Variable(torch.LongTensor(target[i:i + params.batch_size])).cuda()

        # model forward
        #print i,s1_len,s2_len
        output_forward = nli_net((s1_batch, s1_len), (s2_batch, s2_len))
        output_backward = nli_net((s2_batch, s2_len), (s1_batch, s1_len))
        
        soft = torch.nn.Softmax()
        softmax_output_forward = soft(output_forward)
        softmax_output_backward = soft(output_backward)

        sz_forward = softmax_output_forward.size()[0]
        sz_backward = softmax_output_backward.size()[0]

        #pred_forward = output_forward.data.max(1)#[1]
        #pred_backward = output_backward.data.max(1)#[1]

        #print softmax_output_forward.size(),softmax_output_backward.size()
        for j in range (0,1,1):
            entail_forward.append(softmax_output_forward[j][0])
            entail_backward.append(softmax_output_backward[j][0])
            #print softmax_output_forward[j][0][0],softmax_output_backward[j][0][0]

    return entail_forward,entail_backward

# print("Done Training")
# print("Evaluate Now")
# Run best model on test set.
del nli_net
print("Load Saved Model")
nli_net = torch.load(os.path.join(params.outputdir, params.outputmodelname))
print(nli_net)
print("Done: Load Saved Model")

#print('\nTEST : Epoch {0}'.format(epoch))
#evaluate(1e6, 'valid', True)
entail_forward,entail_backward = evaluate(0, 'test', True)
aF , aB = entail_forward[0].data.cpu().numpy(),entail_backward[0].data.cpu().numpy()
print aF,aB

#sys.exit(0)
#print entail_forward[:5]
#print "********"
#print entail_backward[:5]

# sys_file = os.path.join(params.logdir,params.filename) + ".sys.score"
seg_file = os.path.join(params.logdir) + params.flag + ".seg.score"


#f1 = open(sys_file,'a+')
f2 = open(seg_file,'a+')

sz = len(entail_forward)
score = []
score_true = [float(line) for line in open(params.gsfile, 'r')]

#score_true = [open(params.gsfile)]

# lang_pair = params.lp
# hyppath = params.hyppath
# sys_name = ".".join(os.path.basename(hyppath).split('.')[1:-1])

for i in range(0,sz,1):
    e1_score = entail_forward[i].data.cpu().numpy()#[0]
    e2_score = entail_backward[i].data.cpu().numpy()#[0]
    e_score = (e1_score + e2_score)/2
    score.append(e_score)
    #curr_line = [str(i+1),str(e_score),str(score_true[i]),premise[i],hypothesis[i]]
    #test['s1'][i],test['s2'][i]
    #f2.write('\t'.join(curr_line[0:]) + '\n')

print sz,len(test['s1']),len(score),len(score_true),len(premise),len(hypothesis)
print scipy.stats.pearsonr(score,score_true),scipy.stats.spearmanr(score,score_true)

# avg = np.mean(score)
# curr_line = [params.filename,lang_pair,"newstest2014",sys_name,str(avg)]
# f1.write('\t'.join(curr_line[0:]) + '\n')
print "****************************************************"