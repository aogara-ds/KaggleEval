import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Basic preprocessing
# Fill missing numeric values with the median and categorical values with the mode
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        train_df[column] = train_df[column].fillna(train_df[column].mode()[0])
        if column in test_df:
            test_df[column] = test_df[column].fillna(test_df[column].mode()[0])
    else:
        train_df[column] = train_df[column].fillna(train_df[column].median())
        if column in test_df:
            test_df[column] = test_df[column].fillna(test_df[column].median())

# Encode categorical variables
label_encoders = {}
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(pd.concat([train_df[column], test_df[column]]).astype(str))
        train_df[column] = le.transform(train_df[column].astype(str))
        test_df[column] = le.transform(test_df[column].astype(str))
        label_encoders[column] = le

# Splitting the training data for model training and validation
X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = np.log(train_df['SalePrice'])  # Use log-transformed SalePrice for training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Validate the model
y_pred_val = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {val_rmse}')

# Prepare the test set predictions
test_pred = model.predict(test_df.drop(['Id'], axis=1))
test_pred = np.exp(test_pred)  # Revert the log-transformation

# Create submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_pred})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
