# MACHINE-LEARNING-MODEL-IMPLEMENTATION

**Name**: ADIL SHAIKH

**Company**: CODETECH IT SOLUTIONS

**ID**: CT08DJS

**Domain**: Python Programming

**Duration**:12th December to 12th January

**Mentor**: Neela Santhosh Kumar

# OUTPUT PF THIS PROJECT
![Email Spam-OUTPUT](https://github.com/user-attachments/assets/dc780c9b-fd8f-485c-b018-e1da943a8d60)

# Overview Of The Project

This code is a complete workflow for building a spam detection system using machine learning.
Here's an overview of each step:

# Importing Libraries:

The code begins by importing necessary libraries: numpy for numerical operations, pandas for data manipulation, and warnings to suppress warnings.
It also imports functions from sklearn for model training, feature extraction, and evaluation.

# Loading Data:

The dataset is loaded from a CSV file named "mail_data.csv" into a pandas DataFrame (df).
The first few rows of the DataFrame are displayed using df.head().

# Data Exploration:

The code checks for missing values in the dataset with df.isnull().sum().
It retrieves the shape of the DataFrame with df.shape and provides information about the DataFrame structure using df.info().

# Data Preprocessing:

The 'Category' column, which indicates whether an email is spam or ham, is mapped to numerical values: 'spam' is converted to 0 and 'ham' to 1.
The features (X) are set to the 'Message' column, and the target variable (Y) is set to the 'Category' column.

# Splitting the Data:

The dataset is split into training and testing sets using train_test_split, with 20% of the data reserved for testing.
Feature Extraction:

A TfidfVectorizer is used to convert the text messages into numerical features. 
It transforms the text data into a matrix of TF-IDF features, which helps in representing the importance of words in the documents.
The vectorizer is fitted on the training data and then used to transform both the training and testing data.

# Model Training:

A LogisticRegression model is instantiated and trained on the TF-IDF features of the training data.
Predictions are made on the training data, and the accuracy is calculated and printed.

# Model Evaluation:

Predictions are also made on the test data, and the accuracy of the model on the test set is calculated and printed.

# Building a Predictive System:

The code includes a section for making predictions on new user input. A sample email message is provided, and the model predicts whether it is spam or ham.
The input message is transformed using the same TF-IDF vectorizer, and the model's prediction is printed.

Overall, this code demonstrates a typical machine learning pipeline for text classification, specifically for spam detection in emails. 
It includes data loading, preprocessing, feature extraction, model training, evaluation, and making predictions on new data.
The use of logistic regression as the classification algorithm is a common choice for binary classification tasks.
