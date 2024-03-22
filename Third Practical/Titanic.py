import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


data = pd.read_csv("/home/rushabh/Documents/ML practicals/Third Practical/Titanic.csv")

data.groupby("Pclass")["Survived"].value_counts().unstack().plot(kind="bar")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.title("Survival by Passenger Class")
plt.show()

X = data.drop("Survived", axis=1)  # Features
y = data["Survived"]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

def is_informative(feature):
    correlation = feature.corr(data["Survived"])
    if abs(correlation) > 0.1: 
        return True
    return False

if not is_informative(data["Deck"]):
    X_train = X_train.drop("Deck", axis=1)
    X_test = X_test.drop("Deck", axis=1)

imputer = SimpleImputer(strategy="mean")  # Replace with other strategies if suitable
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

onehot_encoder = OneHotEncoder(sparse_output=False)
sex_encoded = onehot_encoder.fit_transform(X_train[["Sex"]])
alone_encoded = onehot_encoder.fit_transform(X_train[["Alone"]])
X_train = pd.concat([X_train.drop(["Sex", "Alone"], axis=1), pd.DataFrame(sex_encoded), pd.DataFrame(alone_encoded)], axis=1)
sex_encoded_test = onehot_encoder.transform(X_test[["Sex"]])
alone_encoded_test = onehot_encoder.transform(X_test[["Alone"]])
X_test = pd.concat([X_test.drop(["Sex", "Alone"], axis=1), pd.DataFrame(sex_encoded_test), pd.DataFrame(alone_encoded_test)], axis=1)


