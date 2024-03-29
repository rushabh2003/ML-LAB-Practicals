import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


data = pd.read_csv('/home/rushabh/Documents/ML practicals/Practical Four/Titanic.csv')
data.describe(include='all')
data.info()

data.hist(column='Age', by='Survived', figsize=(10, 6))
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.title('Age Distribution by Survival Status')
plt.show()

data.groupby('Pclass')['Survived'].value_counts().unstack().plot(kind='bar', colormap='plasma')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers (Survived/Not Survived)')
plt.title('Survival Rate by Passenger Class')
plt.show()

data.drop('Cabin', axis=1, inplace=True)

imputer = SimpleImputer(strategy='mean')  # Adjust strategy as needed
data = pd.DataFrame(imputer.fit_transform(data))


encoder_sex = OneHotEncoder(sparse=False)
data = pd.concat([data, pd.DataFrame(encoder_sex.fit_transform(data[['Sex']]))], axis=1)
data.drop('Sex', axis=1, inplace=True)

encoder_alone = OneHotEncoder(sparse=False)
data = pd.concat([data, pd.DataFrame(encoder_alone.fit_transform(data[['Alone']]))], axis=1)
data.drop('Alone', axis=1, inplace=True)


data['Pclass'] = data['Pclass'].astype('category').cat.codes

encoder_embarked = OneHotEncoder(sparse=False)
data = pd.concat([data, pd.DataFrame(encoder_embarked.fit_transform(data[['Embarked']]))], axis=1)
data.drop('Embarked', axis=1, inplace=True)


X = data.drop('Survived', axis=1)  # Features
y = data['Survived']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

logreg = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)  # Adjust hyperparameters as needed
dtree = DecisionTreeClassifier(max_depth=3)  # Adjust hyperparameters as needed

logreg.fit(X_train, y_train)
knn.fit(X_train, y_train)
dtree.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
y_pred_knn = knn.predict(X_test)
y_pred_dtree = dtree.predict(X_test)


f1_logreg = f1_score(y_test, y_pred_logreg)
f1_knn = f1_score(y_test, y_pred_knn)
f1_dtree = f1_score(y_test, y_pred_dtree)

print(f"F1-score (Logistic Regression): {f1_logreg:.4f}")
print(f"F1-score (KNN): {f1_knn:.4f}")
print(f"F1-score (Decision Tree): {f1_dtree:.4f}")

