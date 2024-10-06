# Vaccine-Usage-Analysis-and-Prediction

Problem Statement:
This project aims to predict the likelihood of people taking an H1N1 flu vaccine using KNN Model. It involves analyzing a dataset containing various features related to individuals' behaviors, perceptions, and demographics, and building a predictive model to determine vaccine acceptance.
Predict the probability of individuals taking an H1N1 flu vaccine based on their characteristics and attitudes. This can help healthcare professionals and policymakers target vaccination campaigns more effectively.

Technologies used:
1) Python Scripting
2) Pandas
3) Seaborn
4) Matplotlib
5) Plotly
6) Streamlit

Approach
1. Data Collection and Preparation
Dataset: The project uses a dataset containing information on various demographic and medical factors like age, sex, chronic conditions, H1N1 worry, vaccination history, etc.
Preprocessing:
* Handling missing data through imputation or removal.
* Encoding categorical variables (like gender) using label encoding or one-hot encoding.
* Scaling numerical features to ensure they contribute equally to the distance metric in KNN.
* Splitting data into training and testing sets to evaluate model performance.
2. Model Selection: K-Nearest Neighbors (KNN)
Why KNN: KNN is chosen because it is a simple and intuitive algorithm that works well for classification tasks, especially when the decision boundary is non-linear. It classifies new data based on the majority class of its nearest neighbors in the feature space.
Tuning Parameters:
K value: The number of neighbors ('k') is crucial for balancing bias and variance. A small 'k' might lead to overfitting, while a large 'k' could result in underfitting.
Distance Metric: Experimented with Euclidean and Manhattan distances to improve prediction accuracy.
3. Model Tuning
Grid Search: Used grid search cross-validation to tune hyperparameters like:
Optimal 'k' value.
Weighting function (uniform or distance-based).
Distance metric (Euclidean, Manhattan).

4. Model Evaluation
Performance Metrics:
Accuracy: Evaluated the proportion of correctly classified instances.
Precision, Recall, F1-score: Focused on these metrics, especially since the dataset may be imbalanced (i.e., more people not taking the vaccine).
Confusion Matrix: Helped visualize the model's true positives, false positives, true negatives, and false negatives.
ROC Curve: Plotted the Receiver Operating Characteristic (ROC) curve to analyze the model's performance at various threshold levels.
5. Comparison with Other Models
The KNN model’s performance was compared with Logistic Regression to see which one generalizes better for the given data.
While Logistic Regression worked well for linearly separable data, KNN provided better flexibility for non-linear decision boundaries in this case, especially after tuning.
6. Developing Streamlit Application
Frontend Interface: A simple, user-friendly interface was developed using Streamlit, allowing users to input various features such as age, gender, and H1N1 worry level.
Backend Integration: The KNN model is integrated into the app, where it takes user inputs and provides a prediction regarding the likelihood of the individual getting vaccinated.
Visualization: Implemented visualizations like correlation heatmaps and bar plots to help users understand which factors contribute most to the predictions.
Results
Model Performance: After hyperparameter tuning, the KNN model achieved:
Accuracy: 81% on the test set.
F1-score: 0.49 (for the minority class of those who took the vaccine).
Precision and Recall: Reasonable precision and recall for predicting both vaccinated and non-vaccinated individuals.
Insights:
Worry about H1N1 and doctor’s recommendation were among the strongest predictors of vaccine uptake.
Chronic medical condition and age group also showed significant correlation with vaccine behavior.
Application Impact: The Streamlit app provides real-time predictions and allows health organizations to quickly assess the likelihood of vaccination for different population segments. This can inform better-targeted campaigns and resource allocation.
Conclusion
By combining machine learning with interactive visualizations, the project has built a robust system for predicting vaccination behavior. The KNN model performed well after tuning and the Streamlit application offers an accessible interface for end users, helping to optimize vaccination efforts. This approach can be expanded to different vaccines or public health campaigns.   
