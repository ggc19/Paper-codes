# research data and model instruction in the article
Some or all data, models, or code that support the findings of this study are available from the corresponding author upon reasonable request. Here, we present the main model codes for reviewers.

The total dataset is composed of 19608 normal samples and 13272 abnormal samples. The training dataset and testing dataset are 80% and 20% of the entire dataset, respectively. Five-fold cross-validation is selected for optimizing model parameters and verifying the training results in order to avoid overfitting. The testing dataset is used as the unseen data to evaluate the models’ performance. 

The file named 'TFCNN.py' is the time frequency convolutional neural network model
The file named 'FCNN.py' is the frequency convolutional neural network model
The file named 'TCNN.py' is the time convolutional neural network model
The file named 'Classification models.py' is the machine learing models used for comparison
The file 'Feature extraction.m' is the time-frequency spectrogram and the traditional signal features. 

The convolutional neural networks are implemented by keras tool and the backend is tensorflow. Other classification models are implemented by scikit-learn tool and some python packages. Run them in Anaconda3 (Spyder) platform. The signal processing methods are implemented by matlab R2014a.

Corresponding author: Shuming Liu, 
Professor, School of Environment, Tsinghua University
E-mail: shumingliu@tsinghua.edu.cn
