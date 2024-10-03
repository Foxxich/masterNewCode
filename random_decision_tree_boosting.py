from ensemble_base import EnsembleBoosting
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class RandomDecisionTreeBoosting(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=10, max_features=5000):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B

    def fit_predict(self):
        f_test = np.zeros(self.X_test.shape[0])

        for b in range(self.B):
            # Kontrolowana losowość w zakresie głębokości drzewa
            max_depth = 3 + np.random.randint(1, 3) if b % 2 == 0 else 3
            tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
            tree.fit(self.X_train, self.y_train)

            f_test += tree.predict(self.X_test) * (1 / (b + 1))

        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels