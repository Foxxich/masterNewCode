from ensemble_base import EnsembleBoosting
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class AdaptiveFeatureSelectionBoosting(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=10, max_features=5000):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B

    def fit_predict(self):
        selected_features = np.arange(self.X_train.shape[1])
        f_test = np.zeros(self.X_test.shape[0])

        for b in range(self.B):
            # Zamiast losowo wybierać cechy, wybieramy te o największej wariancji
            feature_variances = np.var(self.X_train.toarray(), axis=0)
            important_features = np.argsort(feature_variances)[-int(0.7 * len(selected_features)):]
            
            X_train_subset = self.X_train[:, important_features]
            X_test_subset = self.X_test[:, important_features]

            tree = DecisionTreeRegressor(max_depth=1, random_state=42)
            tree.fit(X_train_subset, self.y_train)

            f_test += tree.predict(X_test_subset) * (1 / (b + 1))

        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels
