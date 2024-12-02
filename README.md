# credit-risk-classification
module-20-challenge

Analysis Overview

Purpose
    The goal of this analysis is to evaluate the performance of a Logistic Regression machine learning model in predicting the creditworthiness of borrowers. Specifically, the analysis aims to determine how accurately the model can predict healthy loans (Class 0) versus high-risk loans (Class 1). This evaluation is conducted using the original dataset and recommendations are provided based on the model’s performance.

Dataset
    The dataset consists of information on 77,536 loans, including the following columns:
        •	loan_size: The size of the loan.
        •	interest_rate: The interest rate associated with the loan.
        •	borrower_income: The income of the borrower.
        •	debt_to_income: The borrower’s debt-to-income ratio.
        •	num_of_accounts: The number of accounts associated with the borrower.
        •	derogatory_marks: Negative marks in the borrower’s credit history.
        •	total_debt: The total debt of the borrower.
        •	loan_status: The target variable, indicating whether a loan is healthy (0) or high-risk (1).

    The features (all columns except loan_status) are used to train the machine learning model, while the loan_status column is the label we aim to predict.

Stages of the Machine Learning Process
    The machine learning process follows a systematic approach to ensure accurate predictions and proper evaluation of the model’s performance:
        1.	Prepare the data: Import the dataset, create a DataFrame, and evaluate its features and structure.
        2.	Separate the data into features and labels: Use loan_status as the label and the remaining columns as features.
        3.	Split the data: Use the train_test_split function to divide the dataset into training and testing subsets.
        4.	Import the machine learning model: Use the LogisticRegression model from SKLearn for analysis.
        5.	Instantiate the model: Set up the Logistic Regression model with appropriate parameters.
        6.	Train the model: Fit the model to the training data.
        7.	Make predictions: Use the testing dataset to predict loan statuses.
        8.	Evaluate the model: Assess the predictions using metrics such as accuracy, a confusion matrix, and a classification report.

Machine Learning Methods Utilized
    Primary Model
        •	Logistic Regression: This model is used to classify loans as healthy or high-risk based on borrower data.
    Supporting Functions
        •	train_test_split: Splits the dataset into training and testing subsets.
        •	Evaluation Metrics:
        •	confusion_matrix: Provides insight into prediction errors.
        •	classification_report: Displays precision, recall, and F1 scores for each class.

Results
    Logistic Regression Model Performance:
        •	Accuracy Score: 0.99
        •	Precision:
            •	Class 0 (Healthy Loans): 1.00
            •	Class 1 (High-Risk Loans): 0.84
        •	Recall:
            •	Class 0 (Healthy Loans): 0.99
            •	Class 1 (High-Risk Loans): 0.94

Summary
    The Logistic Regression model demonstrated strong performance in predicting loan statuses, particularly for healthy loans (Class 0), with near-perfect precision and recall. For high-risk loans (Class 1), the precision was 0.84, indicating the model occasionally misclassified healthy loans as high-risk (false positives). However, the recall of 0.94 shows that the model accurately identified most high-risk loans.

    Given its strong performance, the Logistic Regression model appears to be a reliable tool for predicting loan statuses. However, to address the imbalance in prediction performance between the two classes, the following recommendations should be considered:
        •	Explore resampling techniques to address the class imbalance in the dataset.
        •	Compare the Logistic Regression model against other models, such as Random Forest or Gradient Boosting, to determine if they improve predictive performance.
        •	Validate the model on new datasets to ensure its robustness for real-world applications.

    This analysis concludes that the Logistic Regression model is a strong candidate for identifying credit risk but requires further validation before deployment.
