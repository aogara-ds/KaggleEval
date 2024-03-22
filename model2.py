import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor

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
        if column != 'SalePrice' and column in test_df:  # Ensure 'SalePrice' is not in test_df
            test_df[column] = test_df[column].fillna(test_df[column].median())

# Feature engineering
train_df['TotalSF'] = train_df['1stFlrSF'] + train_df['2ndFlrSF']
test_df['TotalSF'] = test_df['1stFlrSF'] + test_df['2ndFlrSF']

# Encode categorical variables
label_encoders = {}
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        le = LabelEncoder()
        le.fit(pd.concat([train_df[column], test_df[column]]).astype(str))
        train_df[column] = le.transform(train_df[column].astype(str))
        test_df[column] = le.transform(test_df[column].astype(str))
        label_encoders[column] = le

# Prepare data
X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = np.log(train_df['SalePrice'])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'min_samples_leaf': [3, 5]
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Validate the model
y_pred_val = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f'Validation RMSE: {val_rmse}')

# Prepare test set predictions
test_pred = best_model.predict(test_df.drop(['Id'], axis=1))
test_pred = np.exp(test_pred)  # Revert log-transformation

# Create submission file
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_pred})
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
