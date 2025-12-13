import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_scaler(model_path='credit_fraud_model.pkl', scaler_path='scaler.pkl'):
    """
    Load trained model and scaler
    """
    print("Loading model and scaler...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def load_test_data(data_path):
    """
    Load test data
    """
    print("Loading test data...")
    df = pd.read_csv(data_path)
    print(f"Test dataset shape: {df.shape}")
    return df

def preprocess_test_data(df, scaler):
    """
    Preprocess test data
    """
    # Separate features and labels
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Standardize features
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def test_model(model, scaler, test_data_path):
    """
    Test model performance
    """
    # Load test data
    df = load_test_data(test_data_path)
    
    # Preprocess test data
    X_test, y_test = preprocess_test_data(df, scaler)
    
    print("Testing model...")
    
    # Prediction
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate evaluation metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print("\n=== Model Evaluation Results ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix:")
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate Transaction', 'Fraudulent Transaction'], 
                yticklabels=['Legitimate Transaction', 'Fraudulent Transaction'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    # Plot prediction probability distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.5, label='Legitimate Transaction', density=True)
    plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.5, label='Fraudulent Transaction', density=True)
    plt.xlabel('Predicted Fraud Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.show()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }

def predict_new_transaction(model, scaler, transaction_data):
    """
    Predict fraud for a new transaction
    """
    # Standardize input data
    transaction_scaled = scaler.transform([transaction_data])
    
    # Prediction
    prediction = model.predict(transaction_scaled)[0]
    probability = model.predict_proba(transaction_scaled)[0][1]
    
    return prediction, probability

def main():
    # Load model and scaler
    try:
        model, scaler = load_model_and_scaler()
        print("Model and scaler loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: Model file not found - {e}")
        print("Please run the training program to generate model files first.")
        return
    
    # Test model (please replace with your test dataset path)
    test_data_path = "creditcard.csv"  # Please modify according to your actual dataset path
    
    try:
        metrics = test_model(model, scaler, test_data_path)
        print("\nModel testing completed!")
        
        # Example: Predict a new transaction
        print("\n=== New Transaction Prediction Example ===")
        # Using the first record in the test set as an example
        df = pd.read_csv(test_data_path)
        sample_transaction = df.drop('Class', axis=1).iloc[0].values
        
        prediction, probability = predict_new_transaction(model, scaler, sample_transaction)
        actual_label = df['Class'].iloc[0]
        
        print(f"Prediction: {'Fraud' if prediction == 1 else 'Legitimate'}")
        print(f"Fraud probability: {probability:.4f}")
        print(f"Actual label: {'Fraud' if actual_label == 1 else 'Legitimate'}")
        
    except FileNotFoundError:
        print(f"Error: Test data file {test_data_path} not found")
        print("Please ensure you have downloaded the credit card fraud dataset and the file path is correct.")

if __name__ == "__main__":
    main()