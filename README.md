Bank Marketing Campaign Prediction Report
Objective:
The goal of this project is to forecast the performance of a bank marketing campaign using a variety of customer characteristics. The idea is to use classification algorithms to anticipate if a consumer will make a term deposit. This binary classification problem uses machine learning techniques such as data preparation, feature selection, model training, and evaluation.
Data Preprocessing:
The dataset used in this research includes customer information as well as the goal variable, which indicates if they have a term deposit (deposit). Preprocessing included the following steps:
•	Handling Missing Values: To maintain the dataset's quality, any missing data was removed or handled accordingly.
•	Categorical Variable Encoding: To make them acceptable for machine learning models, binary categorical variables were label-encoded, whereas non-binary categorical variables were one-hot encoded.
•	Standardization: StandardScaler was used to standardize numerical features, ensuring that all characteristics contributed equally to the model performance.
Feature Selection:
To determine the most relevant attributes for forecasting campaign success, the following feature selection methods were used:
•	Correlation Analysis: Pearson correlation coefficients were used to determine correlations between characteristics and target variables. Highly linked traits were identified for further investigation.
•	Mutual Information: The mutual information between characteristics and the target variable was calculated to assess the dependencies between variables and find the most important features.
•	SelectKBest (ANOVA F-value): The ANOVA F-value method was used to identify the top ten statistically significant features.

Top Features Identified:
The feature selection procedure discovered several essential features, including:
Age, employment, education, marital status, past earnings, and other customer attributes were found to have a strong correlation and impact on the marketing campaign's success.
Model Training:
Three machine learning models were trained to predict if a consumer subscribed to a term deposit:
1.	Logistic Regression : A linear model that is used to predict binary outcomes. It provides an assessment of the likelihood of success.
2.	Decision tree classifier : A non-linear model that divides data based on the most important features.
3.	Random Forest Classifier: A model that combines many decision trees to improve accuracy and resilience.
Cross-Validation:
 To guarantee dependable performance and prevent overfitting, each model was assessed using five-fold cross-validation. Each model's average accuracy scores were calculated.
Evaluation :
Metrics like accuracy, precision, recall, and F1-score were used to assess the models.
Performance Metrics:
	Logistic Regression:
	Accuracy: 0.85
	Precision: 0.81
	Recall: 0.88
	F1-Score: 0.84
	Decision Tree:
	Accuracy: 0.83
	Precision: 0.79
	Recall: 0.84
	F1-Score: 0.81
	Random Forest:
	Accuracy: 0.87
	Precision: 0.83
	Recall: 0.90
	F1-Score: 0.86
Random Forest was the most successful model for campaign success prediction, as evidenced by its superior accuracy and recall.
Feature Importance and Impact
The Random Forest model and logistic regression coefficients were both used to determine feature relevance.
•	Random Forest Feature Importance:
The model revealed which features had the greatest impact on prediction. Top characteristics such as previous, poutcome, contact, and age were recognized as critical in determining whether a consumer will subscribe to a term deposit.


•	Logistic Regression Coefficients:
Previous connections had a favorable impact, but housing and loans had a negative impact on subscription likelihood.
Conclusion:
	The Random Forest model outperformed the Logistic Regression and Decision Tree classifiers in terms of overall accuracy and recall.
	Previous, poutcome, contact, and age are all key predictors of a successful marketing effort. This implies that customer history, campaign contact outcomes, and age all have a significant impact on the choice to subscribe to a term deposit.
	The project highlights the use of data preprocessing, feature selection, and machine learning to solve a real-world business problem, resulting in actionable insights for optimizing marketing campaign strategies.


