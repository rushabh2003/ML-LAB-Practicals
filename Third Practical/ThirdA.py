import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Create a dataframe
df = pd.DataFrame({
  'Country': ['France', 'Spain', 'Germany', 'Spain', 'Germany', 'France', 'Spain', 'France', 'Germany', 'France'],
  'Age': [44, 27, 30, 38, 40, 35, None, 48, 50, 37],
  'Salary': [72000, 48000, 54000, 61000, None, 58000, 52000, 79000, 83000, 67000],
  'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
})

# Fill missing values
df.fillna({'Age': df['Age'].mean(), 'Salary': df['Salary'].mean(), 'Purchased': 'No'}, inplace=True)

# One-hot encode categorical columns
categorical_columns = ['Country', 'Purchased']
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(df[categorical_columns])
encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())

# Join encoded data with original dataframe
df = pd.concat([df, encoded_df], axis=1)
df.drop(categorical_columns, axis=1, inplace=True)

# Print the encoded dataframe
print(df.to_string())
