import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Heart.csv')

# Print the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Print the sum of null values for each column
print("\nMissing values in each column:")
print(df.isnull().sum())
print("\nData types of each column:")
print(df.dtypes)

# Print the count of zeros in each column
print("\nCount of zeros in each column:")
print((df == 0).sum(axis=0))

# Print the mean age
print("\nMean age of patients:")
print(df['Age'].mean())


TP = 45  # True Positives
FP = 55  # False Positives
FN = 5   # False Negatives
TN = 395 # True Negatives

# Constructing the confusion matrix
y_true = [1] * TP + [0] * TN + [1] * FN + [0] * FP  # Actual values
y_pred = [1] * TP + [0] * TN + [0] * FN + [1] * FP  # Predicted values

# Calculate metrics
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print the confusion matrix and metrics
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F-1 Score: {f1:.2f}")