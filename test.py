import pandas as pd

# Ścieżki do plików
true_labels_file = r'C:\Users\Vadym\Documents\masterNewCode\submit.csv'
predicted_labels_file = r'C:\Users\Vadym\Documents\masterNewCode\submit_hybrid_voting.csv'

# Wczytaj pliki
true_labels = pd.read_csv(true_labels_file)
predicted_labels = pd.read_csv(predicted_labels_file)

# Porównanie etykiet
comparison = true_labels['label'] == predicted_labels['label']

# Obliczanie dokładności (accuracy)
correct_predictions = comparison.sum()
total_predictions = len(comparison)
accuracy = correct_predictions / total_predictions

# Wyświetlanie wyników
print(f'Correct predictions: {correct_predictions}/{total_predictions}')
print(f'Accuracy: {accuracy:.4f}')
