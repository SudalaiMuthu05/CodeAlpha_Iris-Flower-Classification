# Iris-Flower-Classification
import kagglehub

# Download latest version
path = kagglehub.dataset_download("saurabh00007/iriscsv")

print("Path to dataset files:", path)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1: Load the dataset
df = pd.read_csv(path + '/Iris.csv')
print("Dataset Preview:")
print(df.head())

# 2: Data preprocessing
# Drop the 'Id' column
df.drop("Id", axis=1, inplace=True)

# Split data into features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4: Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5: Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

print("Iris Flower Classification Completed.")
