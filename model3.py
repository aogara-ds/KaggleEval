import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Basic preprocessing
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        train_df[column] = train_df[column].fillna(train_df[column].mode()[0])
        if column in test_df:
            test_df[column] = test_df[column].fillna(test_df[column].mode()[0])
    else:
        train_df[column] = train_df[column].fillna(train_df[column].median())
        if column != 'SalePrice' and column in test_df:
            test_df[column] = test_df[column].fillna(test_df[column].median())

# Advanced feature engineering
train_df['TotalSF'] = train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['TotalBsmtSF']
test_df['TotalSF'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['TotalBsmtSF']
train_df['Age'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['Age'] = test_df['YrSold'] - test_df['YearBuilt']

# Encode categorical variables
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(pd.concat([train_df[column], test_df[column]]).astype(str))
        train_df[column] = le.transform(train_df[column].astype(str))
        test_df[column] = le.transform(test_df[column].astype(str))

# Prepare data for feature selection
X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = np.log(train_df['SalePrice'])  # Using log-transformed SalePrice for training
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
selector = SelectFromModel(estimator=RandomForestRegressor()).fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(test_df.drop(['Id'], axis=1))

# Initialize models
models = [
    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_leaf=5, random_state=42),
    RandomForestRegressor(n_estimators=200, random_state=42)
]

# Train models and predict
final_predictions = np.zeros(X_test_selected.shape[0])
for model in models:
    model.fit(X_train_selected, y_train)
    y_pred_val = model.predict(X_val_selected)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f'Validation RMSE: {val_rmse}')
    final_predictions += model.predict(X_test_selected) / len(models)

# Revert log-transformation for final predictions
final_predictions = np.exp(final_predictions)

# Create submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")