import pandas as pd
from sklearn.svm import SVC

# Read CSV file
data = pd.read_csv("train_data.csv")

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
# Calculate the IQR and remove outliers for numeric columns
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]

# Handle NaN values
data = data.dropna()

# Define the mapping of string values to numerical values
education_mapping = {
    'Dip': 1,
    'Ad. Dip': 2,
    'Bach': 3,
    'Mst': 4,
    'Doct': 5,
    'P. Doct': 6
}
housing_mapping = {
    'R': 0.5,
    'N': 0,
    'O': 1
}
Res_mapping = {
    'Accept':1,
    'Reject':0
}
Sex_mapping = {
    'M':1,
    'F':0
}
data['LoE'] = data['LoE'].map(education_mapping)
data['Housing'] = data['Housing'].map(housing_mapping)
data['Res'] = data['Res'].map(Res_mapping)
data['MF'] = data['MF'].map(Sex_mapping)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Car'] = le.fit_transform(data['Car'])

# Normalize the data using min-max scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
data = pd.DataFrame(data_scaled, columns=data.columns)

X_train = data.drop('Res',axis=1) 
y_train = data['Res']
#training model
svm = SVC(C=10, kernel='rbf')
svm.fit(X_train, y_train)

# Read CSV file
data1 = pd.read_csv("test_data_2.csv")
# Define the mapping of string values to numerical values
education_mapping = {
    'Dip': 1,
    'Ad. Dip': 2,
    'Bach': 3,
    'Mst': 4,
    'Doct': 5,
    'P. Doct': 6
}
housing_mapping = {
    'R': 0.5,
    'N': 0,
    'O': 1
}
Sex_mapping = {
    'M':1,
    'F':0
}
data1['LoE'] = data1['LoE'].map(education_mapping)
data1['Housing'] = data1['Housing'].map(housing_mapping)
data1['MF'] = data1['MF'].map(Sex_mapping)

le = LabelEncoder()
data1['Car'] = le.fit_transform(data1['Car'])
# Normalize the data using min-max scaling
scaler = MinMaxScaler()
data1_scaled = scaler.fit_transform(data1)
data1 = pd.DataFrame(data1_scaled, columns=data1.columns)

# List of output labels for each model
y_pred = svm.predict(data1)

# Open a text file in write mode
with open('output_labels.txt', 'w') as file:
    # Iterate through the output labels and write them to the text file
    for label in y_pred:
        file.write(str(label) + '\n')

# Close the text file
file.close()