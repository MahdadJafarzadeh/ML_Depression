# Example codes
In this section, we use some of the defined methods of ML_Depression class to make you familiar how to use them:
```python
#%% Test Section: Choose the file of interest
fname = ("C:/ml_project/scripts/tr90_fp1-M2_fp2-M1.h5")   # REM
# Specify channel
ch = 'fp2-M1'
# Defining the object of ML_Depression class
Object = ML_Depression(fname, ch, fs = 200, T = 30)
# Extract features
X,y            = Object.FeatureExtraction() 
#Feature Selection
#ranks, Feat_selected = Object.FeatSelect_Boruta(X, y, max_depth = 7)
# Define the scoring criteria:
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
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
# Cross-validation using ANN
from keras.wrappers.scikit_learn import KerasClassifier
tic = time.time() 
create_model = Object.ANN_Modelling(X, y, units_h1=22,  input_dim=42, units_h2=11, units_output=1,
                  init = 'uniform', activation = 'relu', optimizer = 'adam',
                  loss = 'binary_crossentropy', metrics = ['accuracy'],
                  h3_status = 'deactive', units_h3 = 50)
model = KerasClassifier(build_fn=create_model, epochs=15, batch_size=10, verbose=1)
# evaluate using 10-fold cross validation
results_ANN = cross_validate(model, X, y, cv=10)

print('Taken time : {} secs'.format(time.time()-tic))

#%% Outcome measures
# Defien required metrics here:
Metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score']
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
#%% Applying Randomized grid search to find the best config. of RF

BestParams_RandomSearch, Bestsocre_RandomSearch ,means, stds, params= Object.RandomSearchRF(X, y,
                        estimator = RandomForestClassifier(), scoring = scoring,
                        n_estimators = [int(x) for x in np.arange(10, 500, 20)],
                        max_features = ['log2', 'sqrt'],
                        max_depth = [int(x) for x in np.arange(10, 100, 30)],
                        min_samples_split = [2, 5, 10],
                        min_samples_leaf = [1, 2, 4],
                        bootstrap = [True, False],
                        n_iter = 100, cv = 10)

#%% Test feature selection methods ##
# PCA
PCA_out                            = Object.FeatSelect_PCA(X, y, n_components = 5)
# Boruta
ranks_Boruta, Feat_selected_Boruta = Object.FeatSelect_Boruta(X, y, max_depth = 7)
# Lasso
Feat_selected_lasso                = Object.FeatSelect_LASSO(X, y, C = 1)
#ANOVA
Feat_selected_ANOVA                = Object.FeatSelect_ANOVA(X,y, k = 80)
#Recruisive
ranks_rec, Feat_selected_rec       = Object.FeatSelect_Recrusive(X, y, k = 20)
#### NOW TEST CLASSIFIERS WITH SELECTED FEATS
results_RF  = Object.RandomForest_Modelling(Feat_selected_Boruta, y, scoring = scoring, n_estimators = 200, cv = 10)


#%% Example save featureset
path = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/'
Object.SaveFeatureSet(X, y, path = path, filename = 'feat42_N3')

#%% Example load features:
X, y= Object.LoadFeatureSet(path = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/features/',
                            fname = 'feat42_N3_fp2-M1', 
                            feats = 'featureset', 
                            labels = 'labels')

#%% Combining some REM and SWS epochs

Object.CombineEpochs(directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/',
              ch = 'fp1-M2', N3_fname  = 'tr90_N3_fp1-M2_fp2-M1',
              REM_fname = 'tr90_fp1-M2_fp2-M1',
              saving = True, fname_save = 'tr90_N3&REM_fp1-M2')

#%% How to save some results?
directory = 'P:/3013080.02/ml_project/scripts/1D_TimeSeries/results/' 
fname = '42feats_N3'
with h5py.File((directory+fname + '.h5'), 'w') as wf:
                # Accuracies
                dset = wf.create_dataset('acc_SVM', results_SVM['test_accuracy'].shape, data = results_SVM['test_accuracy'])
                dset = wf.create_dataset('acc_LR' , results_LR['test_accuracy'].shape, data  = results_LR['test_accuracy'])
                dset = wf.create_dataset('acc_RF' , results_RF['test_accuracy'].shape, data  = results_RF['test_accuracy'])
                dset = wf.create_dataset('acc_xgb', results_xgb['test_accuracy'].shape, data = results_xgb['test_accuracy'])
                # Precision
                dset = wf.create_dataset('prec_SVM', results_SVM['test_precision'].shape, data = results_SVM['test_precision'])
                dset = wf.create_dataset('prec_LR' , results_LR['test_precision'].shape, data  = results_LR['test_precision'])
                dset = wf.create_dataset('prec_RF' , results_RF['test_precision'].shape, data  = results_RF['test_precision'])
                dset = wf.create_dataset('prec_xgb', results_xgb['test_precision'].shape, data = results_xgb['test_precision'])
                # Recall
                dset = wf.create_dataset('rec_SVM', results_SVM['test_recall'].shape, data = results_SVM['test_recall'])
                dset = wf.create_dataset('rec_LR' , results_LR['test_recall'].shape, data  = results_LR['test_recall'])
                dset = wf.create_dataset('rec_RF' , results_RF['test_recall'].shape, data  = results_RF['test_recall'])
                dset = wf.create_dataset('rec_xgb', results_xgb['test_recall'].shape, data = results_xgb['test_recall'])
                # f1-score
                dset = wf.create_dataset('f1_SVM', results_SVM['test_f1_score'].shape, data = results_SVM['test_f1_score'])
                dset = wf.create_dataset('f1_LR' , results_LR['test_f1_score'].shape, data  = results_LR['test_f1_score'])
                dset = wf.create_dataset('f1_RF' , results_RF['test_f1_score'].shape, data  = results_RF['test_f1_score'])
                dset = wf.create_dataset('f1_xgb', results_xgb['test_f1_score'].shape, data = results_xgb['test_f1_score'])
