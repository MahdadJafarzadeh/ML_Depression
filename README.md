# ML_Depression

This is the project folder for ML_Depression project, where we aimed at classifitying Normal vs. depressed groups using EEG specific sleep stages (SWS and REM, in particular).
## FeatureExtraction: 
This is the main method to extract features and then use the following methods of supervised machine learning algorithms to classify epochs.
    
INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        
OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".
