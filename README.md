# Credit Risk Prediction Model

## Overview
This project implements a K-Nearest Neighbors (KNN) classifier to predict credit risk, helping financial institutions assess whether a credit application is likely to be "good" or "bad". The model achieves 72% accuracy on the test set using optimized hyperparameters.

## Features
- Automated data preprocessing pipeline
- Handling of missing values using mode imputation
- Categorical variable encoding
- Feature standardization
- Hyperparameter optimization using GridSearchCV
- Model evaluation with detailed metrics
- Model persistence for production use

## Model Performance
- Test Set Accuracy: 72%
- Precision for Bad Credit (0): 57%
- Precision for Good Credit (1): 75%
- Recall for Bad Credit (0): 30%
- Recall for Good Credit (1): 90%

## Requirements
```
pandas
scikit-learn
joblib
```

## Dataset
The model uses a credit risk dataset with the following features:
- checking_status
- duration
- credit_history
- purpose
- credit_amount
- savings_status
- employment
- installment_commitment
- personal_status
- other_parties
- property_magnitude
- age
- other_payment_plans
- housing
- existing_credits
- job
- num_dependents
- own_telephone
- foreign_worker

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/credit-risk-prediction.git
cd credit-risk-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python credit_prediction.py
```

## Model Details
- Algorithm: K-Nearest Neighbors (KNN)
- Best Parameters:
  - n_neighbors: 14
  - metric: manhattan
  - weights: distance
- Cross-Validation Accuracy: 74.14%

## Saved Model Files
- `credit_risk_model_knn_best.pkl`: Trained KNN model
- `label_encoders_knn_best.pkl`: Label encoders for categorical variables
- `scaler_knn_best.pkl`: Feature scaler

## Future Improvements
1. Feature engineering to improve model performance
2. Implementation of other algorithms (Random Forest, XGBoost)
3. API development for model deployment
4. Cross-validation with different scoring metrics
5. Handling of class imbalance

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
