#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[12]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data
df = pd.read_csv('Amazon_Reviews.csv')

# Add print statement to check if data is loaded correctly
print("First 5 rows of the dataframe:")
print(df.head())

# Preprocessing
df['Quarter'] = pd.to_datetime(df['Date']).dt.to_period("Q").astype(str)
columns_to_drop = ['Date', 'Likes', 'Dislikes']
df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

# Separate features and target variable
X = df.drop('Overall_rating', axis=1)
y = df['Overall_rating']

# Add print statement to check for potential data leakage
print("Columns in X:")
print(X.columns)

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Columns to be one-hot encoded and scaled
categorical_cols = ['Name', 'Place', 'Job_type', 'Department', 'Quarter']
numerical_cols = ['work_life_balance', 'skill_development', 'salary_and_benefits', 'job_security', 'career_growth', 'work_satisfaction']

# Data preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])

# Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestRegressor(n_estimators=100, max_features=1.0, random_state=42))])

# Hyperparameter optimization using RandomizedSearchCV
param_dist = {
    'model__n_estimators': [50, 100, 150],
    'model__max_features': [1.0, 'sqrt', 'log2'],
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=100, n_jobs=-1, cv=5, verbose=2, random_state=42)
search.fit(X_train, y_train)

# Best model
best_model = search.best_estimator_

# Predict on test data
y_pred = best_model.predict(X_test)

# Evaluate the model
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')



# In[5]:


df = pd.read_csv('Amazon_Reviews.csv')
df


# In[13]:


# Make sure the column names match those in the original DataFrame
new_review = pd.DataFrame({
    'Name': ['ExampleName'],
    'Place': ['ExamplePlace'],
    'Job_type': ['ExampleJobType'],
    'Department': ['ExampleDepartment'],
    'work_life_balance': [3.5],
    'skill_development': [4.0],
    'salary_and_benefits': [3.7],
    'job_security': [3.0],
    'career_growth': [3.9],
    'work_satisfaction': [4.2],
    'Quarter': ['2023Q1']  # Assuming the review is for the first quarter of 2023
})

# Use the best model to predict the overall rating for the new review
predicted_rating = best_model.predict(new_review)

print(f"Predicted overall rating for the new review: {predicted_rating[0]}")


# In[ ]:




