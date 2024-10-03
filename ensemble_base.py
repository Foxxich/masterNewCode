import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np

class EnsembleBoosting:
    def __init__(self, train_file, test_file, submit_file, max_features=5000):
        # Załaduj dane
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.submit_data = pd.read_csv(submit_file)
        
        # Zastąp brakujące wartości pustymi ciągami
        self.train_data['content'] = self.train_data['title'] + " " + self.train_data['text']
        self.test_data['content'] = self.test_data['title'] + " " + self.test_data['text']
        self.train_data['content'] = self.train_data['content'].fillna('')
        self.test_data['content'] = self.test_data['content'].fillna('')
        
        # TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.X_train = self.vectorizer.fit_transform(self.train_data['content'])
        self.y_train = self.train_data['label']
        self.X_test = self.vectorizer.transform(self.test_data['content'])

    def save_predictions(self, predictions, output_file):
        # Usuń istniejący plik wynikowy, jeśli istnieje
        if os.path.exists(output_file):
            os.remove(output_file)  # Usunięcie istniejącego pliku
        self.submit_data['label'] = predictions
        self.submit_data.to_csv(output_file, index=False)
        print(f"Wyniki zostaly zapisane do: {output_file}")  # Usuwamy polski znak "ł"


    def evaluate(self, predictions_file, true_labels_file):
        # Wczytaj pliki
        true_labels = pd.read_csv(true_labels_file)
        predicted_labels = pd.read_csv(predictions_file)

        # Porównanie etykiet
        comparison = true_labels['label'] == predicted_labels['label']

        # Obliczanie dokładności (accuracy)
        correct_predictions = comparison.sum()
        total_predictions = len(comparison)
        accuracy = correct_predictions / total_predictions

        # Wyświetlanie wyników
        print(f'Correct predictions: {correct_predictions}/{total_predictions}')
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy
