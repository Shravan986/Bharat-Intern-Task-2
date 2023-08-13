import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Data Preprocessing
# Drop irrelevant columns (e.g., PassengerId, Name, Ticket)
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical variables into numerical representations
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Feature Engineering - create a new feature 'FamilySize' combining 'SibSp' and 'Parch'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Drop 'SibSp' and 'Parch' columns
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Define features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.drop('Cabin', axis=1))
X_test_scaled = scaler.transform(X_test.drop('Cabin', axis=1))

# Model Training
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
