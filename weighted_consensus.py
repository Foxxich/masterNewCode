from ensemble_base import EnsembleBoosting
from sklearn.tree import DecisionTreeRegressor

class WeightedConsensus(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=10, max_features=5000):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B

    def fit_predict(self):
        tree = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(self.X_train, self.y_train)
        
        # Konsensus na podstawie rozwijania/przycinania drzewa
        for _ in range(self.B):
            # Obcinanie/rozszerzanie drzewa
            tree.set_params(max_depth=tree.get_params()['max_depth'] + 1)
            tree.fit(self.X_train, self.y_train)
        
        f_test = tree.predict(self.X_test)
        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels
