## Bank Marketing Campaign Analysis
This project analyzes a bank's marketing campaign using Machine Learning techniques to predict whether a client will subscribe to a term deposit. The project uses feature selection, model training, and evaluation to provide actionable insights into campaign performance.
## Key Features
**Dataset Analysis:**
- Preprocessed the bank marketing dataset to handle missing values and categorical data effectively.
**Feature Selection:**
- Used correlation analysis, mutual information, and ANOVA F-value to identify the most important features.
**Model Training:**
- Trained three models—Logistic Regression, Decision Tree, and Random Forest.
Conducted 5-fold cross-validation to evaluate model performance.
**Evaluation:**
- Assessed models using metrics such as Accuracy, Precision, Recall, and F1-Score.
- Identified key features contributing to predictions using Random Forest importance and Logistic Regression coefficients.
## Dataset
The dataset contains client and campaign-related information. The target variable is y, indicating whether the client subscribed to a term deposit (yes or no).

**Key Attributes:**

- Client Information: Age, job, marital status, education, etc.
- Campaign Data: Contact type, last contact duration, number of contacts, etc.
= Target Variable: y (subscription status).
## Steps to Run the Project
**1. Environment Setup**
Install the required libraries:
pip install -r requirements.txt
Example requirements.txt:
pandas
numpy
matplotlib
seaborn
scikit-learn
**2. Dataset Preparation**
- Download the dataset and place it in the data/ directory.
- Ensure the file is named bank.csv or update the script with the correct file path.
**3. Run the Scripts**
- Feature Selection:
Identifies and visualizes the top features.
Run:
python feature_selection.py
- Model Training:
Trains Logistic Regression, Decision Tree, and Random Forest models.
Run:
python model_training.py
- Evaluation:
Evaluates models and displays metrics like accuracy, precision, recall, and F1-score.
Run:
python evaluation.py
## Key Insights
- **Feature Selection:**
- Top Features by Correlation:

Duration, Age, and Poutcome were highly correlated with the target variable.
- Top Features by Mutual Information:

Duration and previous outcomes were the most important predictors.
- Top Features by ANOVA F-value:

Duration, number of contacts, and job type contributed significantly.
## Model Performance:
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	89.12%	87.45%	85.67%	86.55%
Decision Tree	90.34%	88.56%	87.89%	88.22%
Random Forest	92.45%	91.23%	90.78%	91.00%
Random Forest achieved the highest accuracy and F1-Score, indicating strong performance.
## Feature Importance:
- **Random Forest:**
- Duration, previous outcome, and campaign were the most influential features.
- **Logistic Regression Coefficients:**
- Positive Impact: Duration, job type, education.
- Negative Impact: Default, housing loan.
## Code Overview
- **Feature Selection:**
- Conducted correlation analysis, mutual information, and ANOVA F-value analysis.
- Visualized top features using bar plots.
- **Model Training:**
- Trained Logistic Regression, Decision Tree, and Random Forest models.
- Performed 5-fold cross-validation to ensure robustness.
- **Evaluation:**
- Evaluated models using metrics like Accuracy, Precision, Recall, and F1-Score.
- Identified key features and visualized Random Forest feature importance.
## Future Improvements
- Experiment with ensemble techniques like Gradient Boosting or XGBoost.
- Optimize hyperparameters using GridSearchCV for better performance.
- Implement feature engineering to create additional informative features.
- Deploy the model as an API for real-time predictions.
## Conclusion
The Bank Marketing Campaign Analysis demonstrates the importance of data preprocessing, feature selection, and model evaluation in building effective predictive systems. The Random Forest model performed best, offering valuable insights into factors influencing client subscriptions.

