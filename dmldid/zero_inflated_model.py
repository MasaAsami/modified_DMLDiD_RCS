import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

class ZeroInflatedModel:
    def __init__(self, classifier, regressor):
        """
        Initialize the Zero-Inflated Model with optional parameters for ex. LightGBM.

        """
        
        self.classifier = classifier
        self.regressor = regressor
        
        self.is_fitted = False

    def fit(self, X, y, use_sample_weight=False, n_splits=2):
        """
        Fit the Zero-Inflated Model.
        :param X: Feature matrix (pandas DataFrame or numpy array)
        :param y: Target variable (pandas Series or numpy array)
        :param use_sample_weight: Whether to use sample weights for regression
        :param n_splits: Number of splits for cross-prediction when using sample weights
        """
        # Step 1: Fit the classifier to predict zero vs non-zero
        y_binary = (y == 0).astype(int)  # Binary target: 1 if zero, 0 otherwise

        if use_sample_weight:
            # Cross-predict probabilities to avoid overfitting
            prob_zero = np.zeros(len(y))
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_binary[train_idx], y_binary[val_idx]

                self.classifier.fit(X_train, y_train)
                prob_zero[val_idx] = self.classifier.predict_proba(X_val)[:, 1]

            sample_weight = 1 - prob_zero
        else:
            self.classifier.fit(X, y_binary)
            sample_weight = None

        # Step 2: Fit the regressor on non-zero data or with sample weights
        if use_sample_weight:
            self.regressor.fit(X, y, sample_weight=sample_weight)
        else:
            non_zero_indices = y != 0
            X_non_zero = X[non_zero_indices]
            y_non_zero = y[non_zero_indices]
            self.regressor.fit(X_non_zero, y_non_zero)

        self.is_fitted = True

    def predict(self, X):
        """
        Predict using the fitted Zero-Inflated Model.
        :param X: Feature matrix (pandas DataFrame or numpy array)
        :return: Predicted values (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("The model must be fitted before making predictions.")

        # Step 1: Predict probability of zero
        prob_zero = self.classifier.predict_proba(X)[:, 1]

        # Step 2: Predict non-zero values
        pred_non_zero = self.regressor.predict(X)

        # Combine predictions
        predictions = (1 - prob_zero) * pred_non_zero
        return predictions
    
    def get_params(self, deep=True):
        """
        Return the parameters of the model.
        """
        return {
            "classifier": self.classifier,
            "regressor": self.regressor,
        }

    def set_params(self, **params):
        """
        Set the parameters of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Example Usage
if __name__ == "__main__":
    # Simulated data
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(1000, 5), columns=[f"feature_{i}" for i in range(5)])
    y = np.random.choice([0, 1, 2, 3, 4, 5], size=1000, p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

    # Initialize and train the model
    model = ZeroInflatedModel()
    model.fit(X.values, y, use_sample_weight=True)

    # Predict
    predictions = model.predict(X.values)
    print(predictions)
