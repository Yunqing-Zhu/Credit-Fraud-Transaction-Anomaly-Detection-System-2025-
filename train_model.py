import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For handling imbalanced datasets
from imblearn.combine import SMOTETomek  # Combination of over-sampling and under-sampling methods
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_path):
    """
    Load credit card fraud dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of fraud transactions: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    return df

def preprocess_data(df):
    """
    Data preprocessing
    """
    print("Preprocessing data...")
    
    # Separate features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature standardization (Although random forest is insensitive to feature scaling, it helps with some optimizations)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use SMOTE to handle class imbalance
    print("Using SMOTE to handle imbalanced data...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"Original training set shape: {X_train.shape}")
    print(f"Balanced training set shape: {X_train_balanced.shape}")
    print(f"Fraud transaction ratio in training set: {y_train_balanced.mean():.2f}")
    
    # Additional optimization: Feature selection (optional)
    # In credit card fraud detection, all features are usually important, so we keep all features here
    # But if needed, methods like SelectKBest can be used for feature selection
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, scaler

def optimize_hyperparameters(X_train, y_train):
    """
    Optimize hyperparameters using grid search
    """
    print("Optimizing hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create random forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,  # Use 3-fold cross-validation to save time
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_model(X_train, y_train, use_optimization=True):
    """
    Train random forest model
    """
    print("Training model...")
    
    if use_optimization:
        # Use optimized parameters
        model = optimize_hyperparameters(X_train, y_train)
    else:
        # Use default parameters
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
    
    # Evaluate model performance using cross-validation
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"5-fold cross-validation AUC-ROC scores: {cv_scores}")
    print(f"Mean AUC-ROC score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    print("\nEvaluating model...")
    
    # Prediction
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    print(f"\nAUC-ROC: {auc_roc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print(cm)
    
    # Calculate precision, recall and F1 score (extracted from classification report)
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_roc_curve(y_test, y_pred_proba):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def feature_importance_analysis(model, feature_names):
    """
    Analyze feature importance
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nFeature importance ranking:")
    for i in range(min(10, len(feature_names))):  # Display top 10 most important features
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def main():
    # Load data (Note: Please replace the path with your dataset path)
    # Usually the credit card fraud dataset has a file named creditcard.csv after download
    data_path = "..\\creditcard.csv"  # Please modify according to your actual dataset path
    
    try:
        df = load_data(data_path)
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Train model
        model = train_model(X_train, y_train, use_optimization=True)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Plot ROC curve
        plot_roc_curve(y_test, metrics['y_pred_proba'])
        
        # Feature importance analysis
        feature_importance_analysis(model, df.drop('Class', axis=1).columns)
        
        # Save model and scaler
        joblib.dump(model, 'credit_fraud_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("\nModel and scaler saved!")
        
    except FileNotFoundError:
        print(f"Error: Data file {data_path} not found")
        print("Please ensure you have downloaded the credit card fraud dataset and the file path is correct.")
        print("If you used kagglehub to download the dataset, please check the download path.")

if __name__ == "__main__":
    main()
