This repository contains a Jupyter notebook that implements a Support Vector Machine (SVM) model to classify Alzheimer's Disease stages using the OASIS dataset. The notebook includes data preprocessing, model training, hyperparameter tuning, and evaluation.

Overview
The project focuses on:

Data cleaning and preprocessing of the OASIS dataset

Training an SVM model to classify Clinical Dementia Rating (CDR) scores

Experimenting with different kernel types and hyperparameters

Evaluating model performance using accuracy and classification reports

Key Features
Data Preprocessing: Handles missing values, scales features, and encodes categorical variables.

SVM Implementation: Uses scikit-learn's SVC with various kernels (linear, rbf, poly, sigmoid).

Hyperparameter Tuning: GridSearchCV for optimizing kernel type, C, gamma, and degree parameters.

Evaluation Metrics: Accuracy, precision, recall, and F1-score for multi-class classification.

Results
The best performing experiment achieved:

Accuracy: ~83%

Best Parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}

Requirements
Python 3.x

Jupyter Notebook

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

Usage
Clone the repository

Open the notebook in Jupyter

Run cells sequentially to:

Load and preprocess data

Train and evaluate the SVM model

Perform hyperparameter tuning experiments

Future Work
Address class imbalance in the dataset

Experiment with other classification algorithms

Explore feature engineering techniques

Implement more sophisticated evaluation metrics
