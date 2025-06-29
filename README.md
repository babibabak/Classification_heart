# Classification_heart
A machine learning project for predicting heart disease risk using a dataset with features like age, sex, chest pain type, and cholesterol levels. The project implements K-Nearest Neighbors (KNN), Decision Tree, and Logistic Regression models
Heart Disease Prediction
## Overview
This project focuses on predicting the risk of heart disease using a dataset with 303 patient records. It involves data preprocessing, exploratory data analysis (EDA), and the application of multiple machine learning models, including K-Nearest Neighbors (KNN), Decision Tree, and Logistic Regression. Model performance is evaluated using accuracy, Jaccard score, confusion matrix, and classification reports, with the Logistic Regression model achieving an accuracy of 90% and a Jaccard score of 0.85 on the test set.

## Features
- Data Preprocessing: Loading and inspecting the dataset, splitting it into training (80%) and testing (20%) sets, and preparing features for modeling.
- Exploratory Data Analysis (EDA): Visualizing feature distributions and relationships using Seaborn and Matplotlib to understand patterns in the data.
- Machine Learning Models:
    - K-Nearest Neighbors (KNN): Configured for classification of heart disease risk.
    - Decision Tree: Implemented with entropy criterion and a maximum depth of 4, achieving an accuracy of 85.2%.
    - Logistic Regression: Configured with C=0.01 and lbfgs solver, achieving an accuracy of 90% and a Jaccard score of 0.85.
- Model Evaluation: Metrics include accuracy, Jaccard score, confusion matrix, and detailed classification reports (precision, recall, F1-score).
- Visualization: Confusion matrix visualization for Logistic Regression to assess model performance.

## Dataset
The dataset (heart.csv) contains 303 records with 14 features, including:

- `age`: Age of the patient
- `sex`: Sex (1 = male, 0 = female)
- `cp`: Chest pain type
- `trtbps`: Resting blood pressure
- `chol`: Cholesterol level
- `fbs`: Fasting blood sugar
- `restecg`: Resting electrocardiographic results
- `thalachh`: Maximum heart rate achieved
- `exng`: Exercise-induced angina
- `oldpeak`: ST depression induced by exercise
- `slp`: Slope of the peak exercise ST segment
- `caa`: Number of major vessels
- `thall`: Thalassemia
- `output`: Target variable (1 = heart disease, 0 = no heart disease)
The dataset is balanced, with 165 positive (heart disease) and 138 negative cases.

## Requirements
- Python 3.11
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pydotplus`

## Installation

1.Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```

3.Run the Jupyter Notebook:
```bash
jupyter notebook heart.ipynb
```
## Usage
- Load and preprocess the dataset (heart.csv).
- Perform EDA to visualize feature distributions and correlations.
- Train and evaluate KNN, Decision Tree, and Logistic Regression models.
- Analyze model performance using accuracy, Jaccard score, and confusion matrix.

## Model Performance
- K-Nearest Neighbors (KNN): Not fully detailed in the notebook but implemented for comparison.
- Decision Tree:
    - Accuracy: 85.2%


- Logistic Regression:
  - Accuracy: 90%
  - Jaccard Score: 0.85
  - Confusion Matrix: [[35, 1], [5, 20]] (True Positives: 35, True Negatives: 20)
  - Classification Report:
    - Precision: 0.95 (class 0), 0.88 (class 1)
    - Recall: 0.80 (class 0), 0.97 (class 1)
    - F1-Score: 0.87 (class 0), 0.92 (class 1)

## Future Improvements
- Optimize hyperparameters for KNN and Logistic Regression using grid search.
- Explore feature engineering to enhance model performance.
- Implement additional models like Random Forest or Gradient Boosting.
- Expand EDA with more advanced visualizations for deeper insights.
