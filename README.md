# CrediSure - Loan Approval System

Build an end to end **Supervised ML Pipeline** using **kNN, Logistic Regression and Naive Bayes** to predict loan approval.
Implemented **Binary Classification** along with **EDA, Feature Engineering, Model Evaluation ( Precision, Recall and F1 Scores)**. 
Also deployed by creating simple web interface by **Streamlit**. <br>
CrediSure is an **machine learning project** that predicts whether a loan application should be approved based on applicant financial and demographic details.

## Features
- Data preprocessing (imputation, encoding)
- Streamlit-based web application
- Exploratory Data Analysis
- Feature Engineering
- Model Evaluation (Selected the best one)
- Probability-based loan approval decision

## Tech Stack
- Python
- scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborm
- Streamlit

## Model Evaluation Results and Selection
Multiple classification algorithms were evaluated to select the most suitable model for loan approval prediction.
- Models Tested:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes (Gaussian & Bernoulli)

Each model was evaluated with and without feature engineering using standard metrics: Accuracy, Precision, Recall, F1-score.

**Naive Bayes was selected as the final model because:**
- It achieved the highest **Pricision** (less frequent False-Negative - best for loan approvals ) and **F1-score** before feature engineering
- It showed stable and consistent performance compared to other models
- It handled mixed numerical and categorical data effectively
- Naive Bayes Model Evaluation Scores: 
Precision:  90.4 %
Recall: 93.4 %
F1 score:  91.9 %
Accuracy:  94.7 %
CM:  [[123  , 6]
 [  4 , 57]]

**Gaussian Naive Bayes is computationally efficient and suitable for real-time predictions
given these advantages, Gaussian Naive Bayes was chosen for deployment in the Streamlit application.**

## How to Run the Project 
1. Clone the repository:
git clone https://github.com/sonakshigupta29/CrediSure---Loan-Approval-System.git 
3. Install required libraries: pip install streamlit, pandas, numpy, scikit-learn and etc.
4. Run the Streamlit app: streamlit run app.py
The application will open in your browser at:
http://localhost:8501
