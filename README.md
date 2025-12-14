# Credit Card Fraud Detection Model

A credit card fraud detection model based on the Random Forest algorithm.
This project applies machine learning techniques to identify potentially fraudulent transactions from transaction data.

## Project Structure

```
card/
├── train_model.py          # Script for training the model
├── test_model.py           # Script for evaluating the trained model
├── run_all.py              # End-to-end execution script
├── download_dataset.py        # Script for downloading the dataset
├── requirements.txt        # List of required dependencies
└── README.md               # Project documentation
```

## Key Features

- **Random Forest Model Training**：Trains a credit card fraud detection model using the Random Forest algorithm.
- **Data Preprocessing**：Includes feature standardization and handling class imbalance using SMOTE.
- **Hyperparameter Optimization**：Performs hyperparameter tuning with Grid Search.
- **Model Evaluation**：Computes evaluation metrics such as Precision, Recall, F1-score, and AUC-ROC.
- **Visualization and Analysis**：Generates a confusion matrix, ROC curve, and predicted probability distribution plots.
- **Model Persistence**：Supports saving and loading trained models and preprocessing objects.

## Environment Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the Dataset

```python
import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Dataset file path:", path)
```

Please ensure that the `creditcard.csv` data file is available in the current directory.

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This process includes the following steps:
- Loading the dataset
- Data preprocessing (standardization and class imbalance handling)
- Hyperparameter optimization
- Model training
- Model evaluation
- Saving the trained model and the scaler
  
### 2. Test the Model

```bash
python test_model.py
```

This will load the trained model and perform the following:
- Loading the test dataset
- Model evaluation
- Generating evaluation metrics and visualizations
- Providing example predictions for new transactions
  
### 3. Run the End-to-End Script

```bash
python run_all.py
```

This script provides an interactive interface that can:
- Display usage instructions
- Check environment dependencies
- Verify the presence of required data files
- Run the full pipeline (training + testing)
- Run training or testing independently

## Evaluation Metrics

- **Precision**：The proportion of transactions predicted as fraudulent that are actually fraudulent.
- **Recall**：The proportion of actual fraudulent transactions that are correctly identified.
- **F1-score**：The harmonic mean of Precision and Recall.
- **AUC-ROC**：The area under the ROC curve, measuring the model’s ability to distinguish between fraudulent and non-fraudulent transactions.
  
## Optimization Strategies

1. **Handling Class Imbalance**：Applies SMOTE (Synthetic Minority Oversampling Technique) for oversampling the minority class.
2. **Hyperparameter Optimization**：Uses Grid Search to optimize key Random Forest parameters.
3. **Cross-Validation**：Evaluates model performance using 5-fold cross-validation.
4. **Feature Standardization**：Applies standardization to numerical features.
5. **Class Weight Balancing**：Sets balanced class weights within the model to further address class imbalance.

## Dataset Description

The credit card fraud detection dataset contains anonymized credit card transaction data, including both legitimate and fraudulent transactions. The objective is to train a model capable of identifying potentially fraudulent transactions.

- Features: V1–V28 (anonymized features), Time (transaction time), Amount (transaction amount)
- Label: Class (0 = legitimate transaction, 1 = fraudulent transaction)
