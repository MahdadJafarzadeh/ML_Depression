# ML_Depression

This is the project folder for ML_Depression project, where we aimed at classifitying Normal vs. depressed groups using EEG specific sleep stages (SWS and REM, in particular).

    INPUTS: 
        1) filename : full directory of train-test split (e.g. .h5 file saved via Prepare_for_CNN.py)
        2) channel  : channel of interest, e.g. 'fp2-M1'
        3) T        : window size (default is 30 secs as in sleep research)
        4) fs       : sampling frequency (Hz)
        
## FeatureExtraction(): 
This is the main method to extract features and then use the following methods of supervised machine learning algorithms to classify epochs.
    
    INPUTS: 
            It uses global inputs from ML_Depression class, so, doesn't require additional inputs.
        
    OUTPUTS:
        1) X        : Concatenation of all featureset after random permutation.
        2) y        : Relevant labels of "X".

