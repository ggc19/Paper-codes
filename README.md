# Model instruction
Due to our agreement with the research funding agency that the raw data would remain confidential and would not be shared. 

The total dataset is composed of 21334 normal samples and 18102 abnormal samples. The training dataset, validation dataset and testing dataset are 60%, 20% and 20% of the entire dataset, respectively. Five-fold cross-validation is selected for optimizing model parameters and verifying the training results in order to avoid overfitting. The testing dataset is used as the unseen data to evaluate the models’ performance. 

The file named 'TFCNN.py' is the time frequency convolutional neural network model
The file named 'FCNN.py' is the frequency convolutional neural network model
The file named 'TCNN.py' is the time convolutional neural network model
The file named 'Classification models.py' is the machine learing models used for comparison
The file 'Feature extraction.m' is the 3D time frequency spectrogram and the traditional signal features. 
Here, we present the main model codes for reviewers and other researchers.

The convolutional neural networks are implemented by keras tool and the backend is tensorflow. Other classification models are implemented by scikit-learn tool and some python packages. Run them in anaconda 3 platform. The the signal processing methods are implemented by matlab 2014a.
