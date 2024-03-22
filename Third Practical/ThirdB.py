import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns  # Assuming you have seaborn installed
import matplotlib.pyplot as plt
df = pd.read_csv('/home/rushabh/Documents/ML practicals/Third Practical/Day3B.csv')
# Assuming your CSV is loaded into a dataframe named 'df'
print("**1. Display the dataset (head & tail):**")
print(df.head())
print(df.tail())

# Identify dependent and independent variables based on your analysis goals

print("**3. Change index to client_id (assuming it's a unique identifier):**")
df.set_index('client_id', inplace=True)

print("**4. Shape and info of data:**")
print(df.shape)
print(df.info())

print("**5. Show datatype:**")
print(df.dtypes)

print("**6. Convert Datatype:**")
# Convert load_id to object datatype (if necessary)
#   Assuming it's currently a string or numeric
# if not pd.api.types.is_string_dtype(df['load_id']):
#     df['load_id'] = df['load_id'].astype(str)

# Convert repaid to object/categorical datatype (if necessary)
#   Depending on how it's currently encoded
if not pd.api.types.is_categorical_dtype(df['repaid']):
    df['repaid'] = df['repaid'].astype('category')

# Convert loan_start & loan_end to datetime
df['loan_start'] = pd.to_datetime(df['loan_start'])
df['loan_end'] = pd.to_datetime(df['loan_end'])

print("**7. Describe Data:**")
print(df.describe())  # Summary statistics for numeric columns
print(df.describe(include='all'))  # Summary statistics for all columns

print("**8. Plot Loan Amount and Rate as Box Plot:**")
sns.boxplot(
    x = "loan_amount",
    y = "rate",
    showmeans=True,  # Display means as well
    data=df
)
plt.show()

print("**9. Create a new dataframe of Loan amount and rate:**")
loan_rate_df = df[['loan_amount', 'rate']]

print("**10. Transform with StandardScaler:**")
scaler = StandardScaler()
scaled_df = scaler.fit_transform(loan_rate_df)

# scaled_df is the transformed data you can use for further analysis

print("**11. One-Hot Encode loan_type Attribute:**")
encoder = OneHotEncoder()

# Assuming X is your input data
# Use .toarray() or .todense() to get a dense array when transforming the data
dense_array = encoder.fit_transform(df).toarray()

# Or, set the `sparse_output` parameter to False in the method call
dense_array = encoder.fit_transform(df)