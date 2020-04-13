# -*- coding: utf-8 -*-
"""
Created on 12/04/2020
By: Mahdad

Instruction: This is a script to apply ML_Deperession class. 
"""
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from ML_Depression import ML_Depression

#%% Picking featureset of interest and apply classification
Object = ML_Depression(filename='', channel='', fs = 200, T = 30)
path   = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
fname  = 'feat42_lh_REM'
feats  = 'featureset'
labels = 'labels'
X, y   = Object.LoadFeatureSet(path, fname, feats, labels)

#Feature Selection
ranks, Feat_selected = Object.FeatSelect_Boruta(X, y, max_depth = 7)
#X = Feat_selected
# Define the scoring criteria:
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision': make_scorer(precision_score),
           'recall  ' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}   
# Cross-validation using SVM
results_SVM = Object.KernelSVM_Modelling(X, y, scoring = scoring, cv = 10, kernel = 'rbf')
# Cross-validation using logistic regression
results_LR  = Object.LogisticRegression_Modelling(X, y, scoring = scoring, cv = 10)
# Cross-validation using logistic Random Forests
results_RF  = Object.RandomForest_Modelling(X, y, scoring = scoring, n_estimators = 200, cv = 10)
# Cross-validation using XGBoost
results_xgb = Object.XGB_Modelling(X, y, n_estimators = 250, cv = 10, 
                                      max_depth = 3,learning_rate = .1,
                                      scoring = scoring)

#%% Outcome measures
# Defien required metrics here:
Metrics = ['test_accuracy', 'test_precision', 'test_recall  ', 'test_f1_score']
for metric in Metrics:
    #RF
    r1      = results_RF[metric].mean()
    std1    = results_RF[metric].std()
    print(f'{metric} for RF is: {round(r1*100, 2)}+- {round(std1*100, 2)}')
    # xgb
    r2      = results_xgb[metric].mean()
    std2    = results_xgb[metric].std()
    print(f'{metric} for xgb is: {round(r2*100, 2)}+- {round(std2*100, 2)}')
    # SVM
    r3      = results_SVM[metric].mean()
    std3    = results_SVM[metric].std()
    print(f'{metric} for SVM is: {round(r3*100, 2)}+- {round(std3*100, 2)}')
    # LR
    r4      = results_LR[metric].mean()
    std4    = results_LR[metric].std()
    print(f'{metric} for LR is: {round(r4*100, 2)}+- {round(std4*100, 2)}')