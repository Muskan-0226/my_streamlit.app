import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from collections import Counter
from imblearn.over_sampling import SMOTE

# Load and examine the dataset
data = pd.read_csv(r"C:\Users\gmusk\OneDrive\Desktop\user_behavior_dataset.csv")
print(data.head())
print(data.shape)
print(data.isnull().sum())
print(data.dtypes)

# Exploratory Data Analysis (EDA)
# Visualizing the distribution of the target variable
sns.countplot(x='User Behavior Class', data=data)
plt.title('Distribution of User Behavior Class')
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(10, 8))
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Handle missing values only for numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Label encoding for categorical features
label_encode_columns = ['Device Model', 'Operating System', 'Gender']
label_encoder = LabelEncoder()
for column in label_encode_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 
                     'Battery Drain (mAh/day)', 'Number of Apps Installed', 
                     'Data Usage (MB/day)', 'Age']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Split the data into features and target variable
X = data.drop(columns=['User Behavior Class'])
y = data['User Behavior Class']

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution in the training set
print("Class distribution in the training set:", Counter(y_train))

# Handle class imbalance using SMOTE if necessary
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define models with class weights if necessary (for imbalanced classes)
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Cross-validation and hyperparameter tuning for each model
for name, model in models.items():
    print(f"Training {name}...")
    
    # Define a set of hyperparameters to tune for each model
    param_grid = {}
    
    if isinstance(model, LogisticRegression):
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
    elif isinstance(model, RandomForestClassifier):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
        }
    elif isinstance(model, GradientBoostingClassifier):
        param_grid = {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7]
        }
    
    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    # Fit the grid search model with cross-validation on resampled data
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    # Get best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # Evaluate performance metrics
    print(f"{name} Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1-Score:", f1_score(y_test, y_pred, average='weighted'))
    
    # Show Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix to see where the model is failing
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    print("-" * 30)

    # Feature importance visualization for Random Forest if applicable
    if isinstance(best_model, RandomForestClassifier):
        importances = best_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        importance_df.sort_values(by='Importance', ascending=False).head(10).plot(kind='bar', x='Feature', y='Importance')
        plt.title(f'Feature Importances - {name}')
        plt.show()
