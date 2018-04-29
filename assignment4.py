#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 14:33:59 2018

@author: owenwhitley
"""
#import os
import numpy as np
import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

### IMPORT DATA ###
#os.chdir('./mmg1012/assignment_4b')
DMS_summary_onehot = np.loadtxt(fname = 'DMS_summary_onehot.csv', delimiter = ',', dtype = str)
DMS_summary_onehot_df = pd.DataFrame( DMS_summary_onehot[1:,:], columns = DMS_summary_onehot[0,:])
protein_names = (DMS_summary_onehot_df['protein_name'].values)
protein_names.astype(str)
unique_prot_names = np.unique(protein_names)

## make the full matrix of input data (i.e. wt + mutant aa status)
colnames = DMS_summary_onehot_df.columns.tolist()
my_re = re.compile('aa_is')
col_matches = [my_re.search(x) for x in colnames]
col_ind = np.array([x != None for x in col_matches])
colnames_matched = np.array(colnames)[col_ind]
inp_mat_full_df = DMS_summary_onehot_df.loc[:,colnames_matched]
inp_mat_full = inp_mat_full_df.values.astype(int)

## make the full matrix (vector) of DMS_Score values
DMS_score_full = DMS_summary_onehot_df['DMS_score'].values.astype('float')


### Part 1

## store score variables
score_list = []
history_list = []
model_list = []
## perform training with hold out CV holding out 1 protein at a time
for prot_hold_out in unique_prot_names:
    
    ## subset training examples
    rows_use = protein_names != prot_hold_out
    inp_mat_train = inp_mat_full[rows_use]
    DMS_score_train = DMS_score_full[rows_use]
    
    ## subset valitation examples
    rows_val = protein_names == prot_hold_out
    inp_mat_val = inp_mat_full[rows_val]
    DMS_score_val = DMS_score_full[rows_val]
    
    ## create sequential model
    model = Sequential() 
    model.add(Dense(20, activation='relu', input_dim=40))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    # For a binary classification problem
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics = ['accuracy'])
    batch_size_train = np.divide(inp_mat_train.shape[0], 10).astype(int)
    history = model.fit(inp_mat_train, DMS_score_train, epochs=100, batch_size = batch_size_train)
    score = model.evaluate(inp_mat_val, DMS_score_val, batch_size = batch_size_train)
    
    score_list.append(score)
    history_list.append(history)
    model_list.append(model)
    del model
    del score
    del history

my_dict = {}
my_dict['model'] = model_list
my_dict['prot_hold_out'] = prot_hold_out
my_dict['history'] = history_list
my_dict['val_score'] = score_list

CV_results_df = pd.DataFrame(my_dict)