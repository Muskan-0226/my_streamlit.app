import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def load_data():
    choice = input("Would you like to (1) input details manually or (2) upload a dataset? Enter 1 or 2: ")
    
    if choice == '1':
        # Manual input of user details
        device_model = input("Enter Device Model: ")
        operating_system = input("Enter Operating System: ")
        gender = input("Enter Gender: ")
        app_usage_time = float(input("Enter App Usage Time (min/day): "))
        screen_on_time = float(input("Enter Screen On Time (hours/day): "))
        battery_drain = float(input("Enter Battery Drain (mAh/day): "))
        num_apps_installed = int(input("Enter Number of Apps Installed: "))
        data_usage = float(input("Enter Data Usage (MB/day): "))
        age = int(input("Enter Age: "))
        
        # Create a DataFrame from the input data
        data = pd.DataFrame({
            'Device Model': [device_model],
            'Operating System': [operating_system],
            'Gender': [gender],
            'App Usage Time (min/day)': [app_usage_time],
            'Screen On Time (hours/day)': [screen_on_time],
            'Battery Drain (mAh/day)': [battery_drain],
            'Number of Apps Installed': [num_apps_installed],
            'Data Usage (MB/day)': [data_usage],
            'Age': [age]
        })
        
    elif choice == '2':
        # Upload dataset from a CSV file
        file_path = input("Enter the path to your CSV file: ")
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
            
    else:
        print("Invalid choice. Please enter 1 or 2.")
        return None
    
    return data

# Load and examine the dataset
data = load_data()
if data is None:
    exit()

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

# Define and train the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=500),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    
    # Define and tune hyperparameters for each model (example param grid)
    param_grid = {
        # Example parameters; adjust based on your needs.
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
        'Gradient Boosting': {'n_estimators': [100], 'learning_rate': [0.01, 0.1]},
    }.get(name, {})
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_model = grid_search.best_estimator_
    
    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
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

# Save the best-performing model
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
