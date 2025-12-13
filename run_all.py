"""
Credit Card Fraud Detection Comprehensive Run Script

This script provides a complete model training and testing workflow, as well as usage instructions.
"""

import os
import sys

def display_usage():
    """
    Display usage instructions
    """
    usage_text = """
=== Credit Card Fraud Detection Model Usage Instructions ===

1. Environment preparation:
   Ensure the following Python libraries are installed:
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
   - joblib
   - imbalanced-learn (imblearn)

   Installation command:
   pip install pandas numpy scikit-learn matplotlib seaborn joblib imbalanced-learn

2. Data preparation:
   Ensure the credit card fraud dataset (creditcard.csv) is in the current directory
   This file can be downloaded via kagglehub:
   import kagglehub
   path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
   print("Path to dataset files:", path)

3. Model training:
   Run the training script:
   python train_model.py

4. Model testing:
   Run the testing script:
   python test_model.py

5. This comprehensive script features:
   - Automatic check of environment dependencies
   - Check if dataset exists
   - Run training and testing in sequence
   - Display model performance metrics
   """
    print(usage_text)

def check_dependencies():
    """
    Check required dependencies
    """
    required_packages = [
        'pandas',
        'numpy', 
        'sklearn',
        'matplotlib',
        'seaborn',
        'joblib',
        'imblearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing the following packages: {', '.join(missing_packages)}")
        print("Please run: pip install " + ' '.join(missing_packages))
        return False
    else:
        print("✓ All dependencies installed")
        return True

def check_data_file():
    """
    Check if data file exists
    """
    data_files = ['creditcard.csv', 'creditcard.csv.zip']
    
    for file in data_files:
        if os.path.exists(file):
            print(f"✓ Found data file: {file}")
            return True
    
    print("× Data file 'creditcard.csv' not found")
    print("Please download the dataset first: kagglehub.dataset_download('mlg-ulb/creditcardfraud')")
    return False

def run_training():
    """
    Run model training
    """
    print("\n=== Starting Model Training ===")
    try:
        # Import and run training function
        from train_model import main as train_main
        train_main()
        print("✓ Model training completed")
        return True
    except Exception as e:
        print(f"× Model training failed: {str(e)}")
        return False

def run_testing():
    """
    Run model testing
    """
    print("\n=== Starting Model Testing ===")
    try:
        # Import and run testing function
        from test_model import main as test_main
        test_main()
        print("✓ Model testing completed")
        return True
    except Exception as e:
        print(f"× Model testing failed: {str(e)}")
        return False

def main():
    """
    Main function: Execute complete model training and testing workflow
    """
    print("Credit Card Fraud Detection Model - Comprehensive Run Script")
    print("=" * 60)
    
    while True:
        print("\nPlease select an option:")
        print("1. Display usage instructions")
        print("2. Check environment dependencies")
        print("3. Check data file")
        print("4. Run complete workflow (training + testing)")
        print("5. Run model training only")
        print("6. Run model testing only")
        print("7. Exit")
        
        choice = input("\nPlease enter option (1-7): ").strip()
        
        if choice == '1':
            display_usage()
        elif choice == '2':
            check_dependencies()
        elif choice == '3':
            check_data_file()
        elif choice == '4':
            if check_dependencies() and check_data_file():
                if run_training():
                    run_testing()
            else:
                print("\nPlease resolve dependencies or data file issues first")
        elif choice == '5':
            if check_dependencies() and check_data_file():
                run_training()
            else:
                print("\nPlease resolve dependencies or data file issues first")
        elif choice == '6':
            if check_dependencies():
                run_testing()
            else:
                print("\nPlease resolve dependencies first")
        elif choice == '7':
            print("Exiting program")
            break
        else:
            print("Invalid option, please select again")

if __name__ == "__main__":
    main()
