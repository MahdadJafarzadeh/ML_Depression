# ML_Depression

This is the project folder for ML_Depression project, where we aimed at classifitying Normal vs. depressed groups using EEG specific sleep stages (SWS and REM, in particular). This class comprise feature extraction method, classifiers, and grid/randomized search methods, as described in the following sections.

    INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        3) T        : window size (default is 30 secs as in sleep research)
        4) fs       : sampling frequency (Hz)
## 1. FeatureExtraction(): 
This is the main method to extract features and then use the following methods of supervised machine learning algorithms to classify epochs.
    
    INPUTS: 
            It uses global inputs from ML_Depression class, so, doesn't require additional inputs.
        
    OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".

## 2. Classifiers      
        
### 2.1. RandomForest_Modelling( X, y, n_estimators, cv)
A random forest classifier is made using this function and then a k-fold cross-validation will be used to assess the model classification power.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           n_estimator  : number of trees for Random Forest classifier.
           cv           : Cross-validation order
        
    OUTPUTS:
        1) accuracies_RF: Accuracies derived from each fold of cross-validation.
        
### 2.2. KernelSVM_Modelling(X, y, cv, kernel)

A non-linear SVM model is made using this function and then a k-fold cross-validation will be used to assess the model classification power.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           kernel       : kernel function of SVM (e.g. 'db10').
           cv           : Cross-validation order
        
    OUTPUTS:
        1) accuracies_SVM: Accuracies derived from each fold of cross-validation.
### 2.3. LogisticRegression_Modelling(X, y, cv, max_iter)
A Logistic regression model is made using this function and then a k-fold cross-validation will be used to assess the model classification power.

    INPUTS: 
           X            : Featureset input
           y            : Labels (classes)
           max_iter     : Maximum number of iterations during training.
           cv           : Cross-validation order
        
    OUTPUTS:
        1) accuracies_LR: Accuracies derived from each fold of cross-validation.
        
## 3. Randomized search
This is a method to fine tune the hyper parameters of a model. [sometimes] Randomized search is preferable to Grid search due to Randomly selecting instances among the parameters list, leading to faster computations.

### 3.1. RandomSearchRF(X,y, estimator, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf,bootstrap,n_iter):

    INPUTS: 
           X                : Featureset input
           y                : Labels (classes)
           estimator        : RF estimator
           n_estimators     : a list comprising number of trees to investigate.
           max_features     : Number of features to consider at every split.
           max_depth        : Maximum number of levels in tree.
           min_samples_split: Minimum number of samples required to split a node 
           min_samples_leaf : Minimum number of samples required at each leaf node.
           bootstrap        : Method of selecting samples for training each tree.
           n_iter           : The amount of randomly selecting a set of aforementioned parameters.
        
    OUTPUTS:
        1) BestParams_RandomSearch: using 'best_params_' method.
        2) Bestsocre_RandomSearch : using 'best_score_' method.
## 4. Sample code to use methods of class
```ruby
fname = ("P:/3013080.02/ml_project/scripts/1D_TimeSeries/train_test/tr90_N3&REM_fp2-M1.h5")
ch = 'fp2-M1'
# Defining the object of ML_Depression class
Object = ML_Depression(fname, ch, fs = 200, T = 30)
# Extract features
X,y                  = Object.FeatureExtraction()    
# Cross-validation using SVM
accuracies_SVM       = Object.KernelSVM_Modelling(X, y, cv = 10, kernel = 'rbf')
# Cross-validation using logistic regression
accuracies_LR        = Object.LogisticRegression_Modelling(X, y, cv = 10)
# Cross-validation using logistic Random Forests
accuracies_RF        = Object.RandomForest_Modelling(X, y, n_estimators = 200, cv = 10)
# Applying Randomized grid search to find the best config. of RF
Object.RandomSearchRF(X,y)
'''
