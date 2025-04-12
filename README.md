# brain-diseases-detector-and-age-predictor


This project focuses on predicting Alzheimer’s Disease stages and estimating brain age using MRI scan data and machine learning techniques. It aims to assist in early diagnosis and provide supportive tools for clinical decision-making.

 Project Objectives
Alzheimer’s Disease Detection: Classify brain MRI images into categories such as:

Alzheimer’s

Mild Cognitive Impairment (MCI)

Normal (Control)

Brain Age Prediction: Predict the biological age of a person based on their brain MRI scans.

 Technologies Used
Programming Language: Python

Libraries & Frameworks:

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn, TensorFlow / PyTorch

OpenCV, NiBabel (for neuroimaging data)

XGBoost / LightGBM (for age prediction)

Deep Learning Models:

CNNs (Convolutional Neural Networks)

ResNet / VGG variants for feature extraction

Custom models for both classification and regression tasks

Tools:

Jupyter Notebook / Google Colab

Streamlit (for demo, if applicable)

 Dataset
Source: ADNI, Kaggle, and other public MRI datasets

Format: NIfTI (.nii, .nii.gz) or .png (preprocessed)

Preprocessing:

Skull stripping

Normalization

Resizing to standard shape

Data augmentation (rotation, flipping, etc.)

 How It Works
Data Loading & Preprocessing:

Load MRI scans

Normalize and resize

Augment data for training stability

Model Training:

Classification model for Alzheimer's stages

Regression model for brain age prediction

Evaluation:

Metrics:

Classification: Accuracy, Precision, Recall, F1-Score

Regression: MAE, RMSE

Cross-validation and testing on unseen data

Visualization:

Confusion matrix for classification

Loss & accuracy plots

Actual vs Predicted age scatter plot


