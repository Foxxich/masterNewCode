from ensemble_base import EnsembleBoosting
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomDecisionTreeBoosting(EnsembleBoosting):
    def __init__(self, train_file, test_file, submit_file, B=20, max_features=5000):
        super().__init__(train_file, test_file, submit_file, max_features)
        self.B = B

    def fit_predict(self):
        f_test = np.zeros(self.X_test.shape[0])
        weights = []
        predictions_list = []
        prev_error = float('inf')

        for b in range(self.B):
            # Używamy 60% danych w każdej iteracji
            bootstrap_indices = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * 0.6), replace=True)
            X_train_bootstrap = self.X_train[bootstrap_indices]
            y_train_bootstrap = self.y_train[bootstrap_indices]

            # Regularization in Logistic Regression
            if b % 2 == 0:
                model = LogisticRegression(max_iter=500, penalty='l2', C=0.01)  # L2 regularization
            else:
                model = LogisticRegression(max_iter=500, penalty='l1', solver='liblinear', C=0.01)  # L1 regularization

            # Alternatywnie możemy użyć RandomForest, aby uzupełnić modele
            if b % 4 == 0:
                model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=b)

            # Trenujemy model
            model.fit(X_train_bootstrap, y_train_bootstrap)

            # Zapisujemy predykcje
            pred_test = model.predict(self.X_test)
            predictions_list.append(pred_test)

            # Obliczamy błąd treningowy
            pred_train = model.predict(X_train_bootstrap)
            train_error = np.mean(np.abs(pred_train - y_train_bootstrap))

            # Ustawiamy wagę modelu
            weight = 1 / (train_error + 1e-5)
            weights.append(weight)

            # Aktualizujemy poprzedni błąd
            prev_error = train_error

        # Uśrednione predykcje z wagami
        predictions_list = np.array(predictions_list)
        final_predictions = np.average(predictions_list, axis=0, weights=weights)
        predicted_labels = (final_predictions >= 0.5).astype(int)
        return predicted_labels
