import numpy as np
from ensemble_base import EnsembleBoosting
from sklearn.tree import DecisionTreeRegressor

class WeightedConsensus(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=10, max_features=5000):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B

    def fit_predict(self):
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(self.X_train, self.y_train)
        previous_error = float('inf')
        
        for _ in range(self.B):
            predictions = tree.predict(self.X_train)
            error = np.mean(np.abs(self.y_train - predictions))

            # Zmieniamy głębokość drzewa w bardziej kontrolowany sposób
            if error < previous_error * 0.95:  # Głębsze drzewo tylko przy dużej poprawie
                tree.set_params(max_depth=tree.get_params()['max_depth'] + 1)
            elif error > previous_error * 1.05:  # Zmniejszamy głębokość przy pogorszeniu
                tree.set_params(max_depth=max(2, tree.get_params()['max_depth'] - 1))
                
            previous_error = error
            tree.fit(self.X_train, self.y_train)

        f_test = tree.predict(self.X_test)
        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels

