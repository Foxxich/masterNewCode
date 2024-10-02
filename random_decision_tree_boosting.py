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
            tree = DecisionTreeRegressor(max_depth=3, random_state=42)
            tree.fit(self.X_train, self.y_train)

            # Modyfikowanie losowo reguł podziału w drzewie
            if b % 2 == 0:
                tree.set_params(max_depth=tree.get_params()['max_depth'] + 1)
            else:
                tree.set_params(max_depth=max(1, tree.get_params()['max_depth'] - 1))

            f_test += tree.predict(self.X_test) * (1 / (b + 1))

        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels
