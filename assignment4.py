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
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model as LM
from sklearn import metrics


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
    model.add(Dense(40, activation='relu', input_dim=40))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(20, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics = ['mean_squared_error'])
    batch_size_train = np.divide(inp_mat_train.shape[0], 20).astype(int)
    history = model.fit(inp_mat_train, DMS_score_train, epochs=100, batch_size = batch_size_train)
    score = model.evaluate(inp_mat_val, DMS_score_val, batch_size = batch_size_train)
    score = np.array(score).reshape(1,2)
    score_list.append(score)
#    history_list.append(history)
#    model_list.append(model)
    
    ## write history and results to file
    
    history_df = pd.DataFrame(history.history)
    # score_df = pd.DataFrame(score, columns = ['loss', 'MSE'])
    fname_prefix = 'CV_hold_out_' + prot_hold_out
    fname_history = fname_prefix + '_history.csv'
    # fname_score = fname_prefix + '_score.csv'
    history_df.to_csv(fname_history)
   # score_df.to_csv(fname_score)
    

score_array = np.array(score_list, dtype = float)
score_df = pd.DataFrame(score_array, rows = unique_prot_names, columns = ('Loss', 'MSE'))
fname_score = 'NN_validation_scores.csv'
score_df.to_csv(fname_score)
## train model on whole data
    
model = Sequential() 
model.add(Dense(40, activation='relu', input_dim=40))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics = ['mean_squared_error'])
batch_size_train = np.divide(inp_mat_full.shape[0], 20).astype(int)
history = model.fit(inp_mat_full, DMS_score_full, epochs=100, batch_size = batch_size_train)

## save results to file
history_df = pd.DataFrame(history.history)
fname_history = 'full_set_train_history.csv'
history_df.to_csv(fname_history)


## Make matrix of DMS score predictions for each mutant wt pair possible
## colnames_matched contains mutant AAs as first 20 entries, WT aas as 
## 2nd 20.

## make array to hold all possible combinations of one hot encoded values
test_array = np.zeros((400,40))
## row index for test array to be incremented in loop
row_ind = 0
## make numpy array to hold set of all possible mutation combinations
name_combinations = np.zeros((400,2)).astype(str)

for i in range(0,20):
    
    for j in range(20,40):
        
        ## make the following vector: 20 element vector of one hot encoding
        ## for mutant status for the 20 aas concatenated with another 
        ## 20 element vector of hone hot encoding for WT aa status
        ## this is equivalent to the vector used to train the model
        
        inp_vect_i_j = np.zeros((40,))
        inp_vect_i_j[i] = 1.0
        inp_vect_i_j[j] = 1.0
        
        ## add vector to array
        test_array[row_ind, :] = np.copy(inp_vect_i_j)
        ## add name combination to name mapping array
        name_combinations[row_ind,:] = np.array((colnames_matched[i], colnames_matched[j]), dtype = str)
        ## increment
        row_ind += 1
        del inp_vect_i_j
        
## make predictions for all enumerated mutation combinations
        
predicted_DMS = model.predict(x = test_array, verbose = 1)
DMS_mut_array = np.hstack((name_combinations, predicted_DMS))
DMS_mut_df = pd.DataFrame(DMS_mut_array, columns = ('mut_aa', 'wt_aa', 'predicted_DMS'))


## save to file
fname_DMS_pred = 'DMS_predictions.csv'
DMS_mut_df.to_csv(fname_DMS_pred)

## take predicted values and make a 20x20 matrix

DMS_pred_mat = np.zeros((20,20))
row_ind = 0

for i in range(0,20):
    
    for j in range(0,20):
        
        
        DMS_pred_mat[i, j] = DMS_mut_df['predicted_DMS'][row_ind]
        row_ind += 1

fname_DMS_pred_mat = 'DMS_predictions_matrix_fmt.csv'
np.savetxt(fname_DMS_pred_mat, DMS_pred_mat, delimiter = ',')

## plot matrix as a heatmap

wt_aa = colnames_matched[20:40]
mut_aa = colnames_matched[0:20]
#DMS_pred_mat_transpose = DMS_pred_mat.T

def make_heatmap(mat_for_heatmap, xlabs, ylabs, title = ''):
    mut_aa = xlabs
    wt_aa = ylabs
    fig, ax = plt.subplots()
    # im = ax.imshow(DMS_pred_mat_transpose)
    im = ax.imshow(DMS_pred_mat)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(mut_aa)))
    ax.set_yticks(np.arange(len(wt_aa)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(mut_aa)
    ax.set_yticklabels(wt_aa)
    ax.set_title(title)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    
#    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
#                                norm=norm,
#                                orientation='horizontal')
#    cb1.set_label('Some Units')
    cax = ax.imshow(DMS_pred_mat, interpolation='nearest')
    cbar = fig.colorbar(cax, ticks=[0,1], orientation='vertical')
    cbar.ax.set_yticklabels(['Low', 'High'])  # horizontal colorbar
    fig.tight_layout()
    return(fig)

pred_DMS_fig = make_heatmap(DMS_pred_mat, mut_aa, wt_aa, title = 'predicted DMS scores')

plt.savefig('predicted_DMS_scores_heatmap.png')




plt.show()



## assess correlation of predicted DMS with actual DMS, provean scores and BLOSUM scores
predicted_DMS_data =  model.predict(x = inp_mat_full, verbose = 1)
predicted_DMS_data = predicted_DMS_data.reshape(predicted_DMS_data.shape[0],)
predicted_DMS_data = predicted_DMS_data
corr_DMS = np.corrcoef(x = predicted_DMS_data, y = DMS_score_full)
provean = DMS_summary_onehot_df['provean'].values.astype(float)
corr_provean = np.corrcoef(x = predicted_DMS_data, y = provean)
blosum = DMS_summary_onehot_df['blosum'].values.astype(float)
corr_blosum = np.corrcoef(x = predicted_DMS_data, y = blosum)

corr_array = np.array([corr_DMS[0,1], corr_provean[0,1], corr_blosum[0,1]]).reshape(3,)
np.savetxt('correlations_predicted_DMS.tsv', X = corr_array, delimiter = '\t', header = 'pred_DMS_x_actual_DMS\tpred_DMS_x_provean\tpre_DMS_blosum')
    
    
###############################################################################

## Make a linear model attempting to map WT-> mut pairs to DMS score

score_list_LM = []

for prot_hold_out in unique_prot_names:
    
    ## subset training examples
    rows_use = protein_names != prot_hold_out
    inp_mat_train = inp_mat_full[rows_use]
    DMS_score_train = DMS_score_full[rows_use]
    
    ## subset valitation examples
    rows_val = protein_names == prot_hold_out
    inp_mat_val = inp_mat_full[rows_val]
    DMS_score_val = DMS_score_full[rows_val]
    ## fit model
    regr = LM.LinearRegression()
    regr.fit(inp_mat_train, DMS_score_train)
    ## make predictions, get MSE
    predict_val = regr.predict(inp_mat_val)
    MSE = metrics.mean_squared_error(DMS_score_val, predict_val)
    
    score_list.append(MSE)
    
score_array_LM = np.array(score_list_LM)