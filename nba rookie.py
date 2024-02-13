#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


# Load the dataset
data = pd.read_csv(r'C:\Users\opeol\Desktop\7CSO30 AI ASSESSMENT\nba_rookie_data.csv')


# In[3]:


# Print the column names 
print("Column Names:", data.columns) 


# In[4]:


# Drop 'Name' column from features 
data = data.drop("Name", axis=1) 


# In[5]:


# Define features (X) and target variable (y) 
X = data.drop("TARGET_5Yrs", axis=1) 
y = data["TARGET_5Yrs"]


# In[6]:


# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Impute missing values 
imputer = SimpleImputer(strategy="mean") 
X_train = imputer.fit_transform(X_train) 
X_test = imputer.transform(X_test)


# In[8]:


# Standardize features 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)


# In[ ]:


# Neural Network 
nn_model = MLPClassifier(hidden_layer_sizes=(100,),
                         max_iter=1000) 
nn_model.fit(X_train, y_train) 
nn_predictions = nn_model.predict(X_test)


# In[ ]:


# Evaluate the models 
def evaluate_model(predictions, model_name): 
    accuracy = accuracy_score(y_test, predictions) 
    confusion_mat = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    
    print(f"Results for {model_name}:") 
    print(f"Accuracy: {accuracy:.4f}") 
    print(f"Confusion Matrix:\n{confusion_mat}") 
    print(f"Classification Report:\n{classification_rep}") 
    print("\n") 


# In[ ]:


# Display results 
evaluate_model(logreg_predictions, "Logistic Regression") 
evaluate_model(nb_predictions, "Gaussian Naive Bayes") 
evaluate_model(nn_predictions, "Neural Network")


# In[ ]:




