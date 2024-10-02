from ensemble_base import EnsembleBoosting
from sklearn.tree import DecisionTreeRegressor
import numpy as np

class HybridVotingBoosting(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=10, max_features=5000, max_depth=3):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B  # liczba iteracji
        self.max_depth = max_depth  # Maksymalna głębokość drzewa

    def fit_predict(self):
        weights = np.ones(self.X_train.shape[0])  # Inicjalizacja wag
        classifiers = []
        f_test = np.zeros(self.X_test.shape[0])  # Początkowe przewidywania dla testów
        
        for b in range(self.B):
            # Zwiększamy głębokość drzewa, aby model mógł uchwycić więcej szczegółów
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(self.X_train, self.y_train, sample_weight=weights)
            classifiers.append(tree)

            # Przewidywania dla danych treningowych
            predictions = tree.predict(self.X_train)
            error = np.abs(self.y_train - predictions)  # Błąd
            weights += error  # Aktualizacja wag na podstawie błędu
            
            # Przewidywania dla testów
            f_test += tree.predict(self.X_test) * (1 / (b + 1))  # Dynamiczne ważenie

        predicted_labels = (f_test >= 0.5).astype(int)
        return predicted_labels
