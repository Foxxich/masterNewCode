from ensemble_base import EnsembleBoosting
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class HybridVotingBoosting(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=10, max_features=5000, max_depth=3):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B
        self.max_depth = max_depth

    def fit_predict(self):
        weights = np.ones(self.X_train.shape[0])
        classifiers = []
        f_test = np.zeros(self.X_test.shape[0])
        
        for b in range(self.B):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(self.X_train, self.y_train, sample_weight=weights)
            classifiers.append(tree)

            predictions = tree.predict(self.X_train)
            error = np.abs(self.y_train - predictions)
            error = np.clip(error, 1e-10, 0.5)  # Mniej agresywna aktualizacja
            weights = weights * np.exp(0.5 * error)  # Zmniejszamy tempo aktualizacji wag
            
            f_test += tree.predict(self.X_test) * (1 / (b + 1))

        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels
