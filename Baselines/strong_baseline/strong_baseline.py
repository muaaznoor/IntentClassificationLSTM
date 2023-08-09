# -*- coding: utf-8 -*-
"""strong-baseline.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-gHzyDS9llGBQs-uhgPZowubGE1Gbssf

## Import Packages
"""

from __future__ import division
import os, sys, re, json, time, datetime, shutil
import itertools, collections
from importlib import reload

import random 
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from numpy.linalg import *
np.random.seed(42)  # don't change this line

import base64

# NLTK, NumPy, and Pandas.
import nltk
nltk.download('punkt')
from nltk.tree import Tree
from numpy import random as rd
from nltk.tokenize import word_tokenize
import random

import collections
import re
import time
import itertools
from collections import defaultdict, Counter

import glob
from argparse import ArgumentParser

#Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""## Load [Datasets](https://huggingface.co/datasets/SetFit/amazon_massive_intent_en-US)"""

from google.colab import drive
drive.mount('/content/drive')

df_train = pd.read_json("/content/drive/Shareddrives/CIS-5300_Final-Project/train.jsonl", lines=True)
df_train.shape

df_validation = pd.read_json("/content/drive/Shareddrives/CIS-5300_Final-Project/validation.jsonl", lines=True)
df_validation.shape

df_test = pd.read_json("/content/drive/Shareddrives/CIS-5300_Final-Project/test.jsonl", lines=True)
df_test.shape

"""## Download [Glove Embeddings](https://nlp.stanford.edu/projects/glove/) 

"""

#this takes about 10 minutes to run
#!wget -nc https://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip
!unzip /content/drive/Shareddrives/CIS-5300_Final-Project/glove.840B.300d.zip
!ls -lat

glove_file = "glove.840B.300d.txt"

"""## Preprocess"""

df_all = pd.concat([df_train, df_validation, df_test]).reset_index()

tokenized_data = [word_tokenize(df_all['text'][i]) for i in range(len(df_all['text']))]

vocab = {word for sentence in tokenized_data for word in sentence}
vocab.add('<PAD>')

word_to_idx = { w : i for i, w in enumerate(vocab) }

def pre_process(data, word_to_idx):
  tokenized_data = [word_tokenize(data['text'][i]) for i in range(len(data))]

  lens = np.array([len(sentence) for sentence in tokenized_data])
  
  tokens = [word_to_idx[word] for sentence in tokenized_data for word in sentence]

  padded_tokens = np.full([len(tokenized_data), max(lens)], word_to_idx['<PAD>'])
  for i in range(len(tokenized_data)):
    for j in range(len(tokenized_data[i])):
      padded_tokens[i][j] = word_to_idx[tokenized_data[i][j]]

  labels = np.array(data['label'])
    
  return padded_tokens, lens, labels

padded_tokens, lens, labels = pre_process(df_train, word_to_idx)

"""## Get Glove Embeddings"""

#takes about 1 minute to read through the whole file and find the words we need. 
def get_glove_mapping(vocab, file):
    """
    Gets the mapping of words from the vocabulary to pretrained embeddings
    
    INPUT:
    vocab       - set of vocabulary words
    file        - file with pretrained embeddings

    OUTPUT:
    glove_map   - mapping of words in the vocabulary to the pretrained embedding
    
    """
    
    glove_map = {}
    with open(file,'rb') as fi:
        for l in fi:
            try:
                #### STUDENT CODE HERE ####
                emd_lst = l.decode().split(' ')
                word = emd_lst.pop(0)
                emd_lst = [float(n) for n in emd_lst]

                if word in vocab:
                  glove_map[word] = np.array(emd_lst)

                #### STUDENT CODE ENDS HERE ####
            except:
                #some lines have urls, we don't need them.
                pass
    return glove_map

glove_map = get_glove_mapping(vocab,glove_file)

"""## Get Embedding Matrix"""

def get_dimensions():
    d_out =  60 #number of outputs
    n_embed =  len(vocab) #size of the dictionary of embeddings
    d_embed =  300 # the size of each embedding vector
    return d_out, n_embed, d_embed
d_out,n_embed,d_embed = get_dimensions()

def get_embedding_matrix(n_embed, d_embed, glove_map):
    """
    Initialize the weight matrix
    
    INPUT:
    n_embed         - size of the dictionary of embeddings
    d_embed         - the size of each embedding vector

    OUTPUT:
    embedding_matrix  - matrix of mapping from word id to embedding 
    """
    #### STUDENT CODE HERE ####
    train_words = vocab
    embedding_matrix = np.full((n_embed, d_embed), np.random.normal())

    for i, word in enumerate(train_words):

        if word in glove_map.keys():

            embedding_matrix[i] = glove_map[word]
    
    #### STUDENT CODE ENDS HERE ####
    return embedding_matrix

embedding_matrix = get_embedding_matrix(n_embed, d_embed, glove_map)
embedding_data = (embedding_matrix.shape, embedding_matrix[:155])

"""## Define Embedding Layer"""

def create_emb_layer(embedding_matrix, non_trainable=False):
    """
    Create the embedding layer
    
    INPUT:
    embedding_matrix  - matrix of mapping from word id to embedding
    non_trainable   - Flag for whether the weight matrix should be trained. 
                      If it is set to True, don't update the gradients

    OUTPUT:
    emb_layer       - embedding layer 
    
    """
    #### STUDENT CODE HERE ####

    emb_layer = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), padding_idx=140)

    #### STUDENT CODE ENDS HERE ####

    return emb_layer

"""## Define Dataloader """

class SSTpytorchDataset(Dataset):
    def __init__(self, dataset, tokens, word_to_idx, word_dropout = 0.3, split='train'):
        super(SSTpytorchDataset, self).__init__()
        assert split in ['train', 'test', 'dev'], "Error!"
        self.ds = dataset
        self.split = split
        self.word_to_idx = word_to_idx
        #self.word_dropout = word_dropout
        self.data_x, self.data_ns, self.data_y = pre_process(dataset, self.word_to_idx)

    def __len__(self):
        return self.data_x.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        y = self.data_y[idx] 

        return self.data_x[idx], self.data_ns[idx], y

"""## Define Train & Evaluate Functions"""

def train(model, word_to_idx, lr = .005, drop_out = 0, word_dropout = .3, batch_size = 16, weight_decay = 1e-5, model_type= "LSTM"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainset = SSTpytorchDataset(df_train, word_dropout, word_to_idx, 'train')
    testset = SSTpytorchDataset(df_test, word_dropout, word_to_idx, 'test')
    devset = SSTpytorchDataset(df_validation, word_dropout, word_to_idx, 'dev')

    train_iter = DataLoader(trainset, batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(testset, batch_size, shuffle=False, num_workers=0)
    dev_iter = DataLoader(devset, batch_size, shuffle=False, num_workers=0)
    
    model = model
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay = weight_decay)
    print('check')
    acc, val_loss = evaluate(dev_iter, model, device, model_type)
    best_acc = acc

    print(
        'epoch |   %        |  loss  |  avg   |val loss|   acc   |  best  | time | save |')
    print(
        'val   |            |        |        | {:.4f} | {:.4f} | {:.4f} |      |      |'.format(
            val_loss, acc, best_acc))

    iterations = 0
    last_val_iter = 0
    train_loss = 0
    start = time.time()
    _save_ckp = ''
    for epoch in range(epochs):
        
        n_correct, n_total, train_loss = 0, 0, 0
        last_val_iter = 0
        
        for batch_idx, batch in enumerate(train_iter):
            # switch model to training mode, clear gradient accumulators
            
            model.train()
            optimizer.zero_grad()
            iterations += 1

            data, lens, label = batch
            data = data.to(device)
            label = label.to(device).long()

            answer = model(data, lens)

            loss = criterion(answer, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print('\r {:4d} | {:4d}/{} | {:.4f} | {:.4f} |'.format(
                epoch, batch_size * (batch_idx + 1), len(trainset), loss.item(),
                       train_loss / (iterations - last_val_iter)), end='')

            if iterations > 0 and iterations % dev_every == 0:
                acc, val_loss= evaluate(dev_iter, model, device, model_type)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), save_path)
                    _save_ckp = '*'

                print(
                    ' {:.4f} | {:.4f} | {:.4f} | {:.2f} | {:4s} |'.format(
                        val_loss, acc, best_acc, (time.time() - start) / 60,
                        _save_ckp))

                train_loss = 0
                last_val_iter = iterations
    model.load_state_dict(torch.load(save_path)) #this will be the best model
    test_y_pred = evaluate(dev_iter, model, device, model_type, "test")
    print("\nValidation Accuracy : ", evaluate(dev_iter,model, device, model_type))
    return best_acc, test_y_pred

def evaluate(loader, model, device, model_type = "LSTM", split = "dev"):
    model.eval()
    n_correct, n = 0, 0
    losses = []
    y_pred = []
    labels = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            data, lens, label = batch
            data = data.to(device)
            label = label.to(device).long()
            
            answer = model(data, lens)
            if split != "test":
                n_correct += (torch.max(answer, 1)[1].view(label.size()) == label).sum().item()
                n += answer.shape[0]
                loss = criterion(answer, label)
                losses.append(loss.data.cpu().numpy())
            else:
                y_pred.extend(torch.max(answer, 1)[1].view(label.size()).tolist())
                labels.extend(label.tolist())
    if split != "test":
        acc = 100. * n_correct / n
        loss = np.mean(losses)
        return acc, loss
    else:
        return y_pred

"""## Define LSTM Network"""

import random as random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM, GRU

class LSTM_Classifier(nn.Module):

    def __init__(self,
                 n_embed=20000,
                 d_embed=300,
                 d_hidden=150,
                 d_out=59,
                 embeddings=None,
                 nl = 2,
                 bidirectional = True,
                 gru = False
                 ):
        super(LSTM_Classifier, self).__init__()

        self.d_hidden = d_hidden
        self.bidrectional = bidirectional
        self.num_layers = nl

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embed = create_emb_layer(embedding_matrix,False)
        #### STUDENT CODE STARTS HERE ####
        self.lstm = nn.LSTM(d_embed, self.d_hidden, self.num_layers, batch_first=True,
                              bidirectional=bidirectional)
        
        if bidirectional:
          self.fc_out = nn.Linear(self.d_hidden*2, d_out)
        else:
          self.fc_out = nn.Linear(self.d_hidden, d_out)
        self.dropout = nn.Dropout(p=0.2)
        #### STUDENT CODE ENDS HERE ####

    def forward(self, text, seq_lengths):

        batch_size = text.size()[0]

        #### STUDENT CODE STARTS HERE ####
        #Fill the missing pieces for the forward pass
        x = text
        #print(x)
        x = self.embed(x)
        x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        #x = self.dropout(x)
        x = self.fc_out(x)
        x, _ = torch.max(x, 1)
        #### STUDENT CODE ENDS HERE ####

        return x

"""## Training Loop"""

torch.manual_seed(1234)
criterion = nn.CrossEntropyLoss()
batch_size = 128
epochs = 10
dev_every = 100
lr = 0.01
save_path = "best_model"
drop_out = 0
word_dropout = 0
weight_decay = 0

model = LSTM_Classifier(n_embed=n_embed, d_embed=d_embed, d_hidden=150, d_out=d_out, bidirectional=False)
dev_value, test_y_pred = train(model, word_to_idx, lr, drop_out, word_dropout, batch_size, weight_decay, "LSTM")