import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class KNNClassifier:
    def __init__(self, k=5, weighted=False):
        self.k = k
        self.weighted = weighted
        self.scaler = StandardScaler()
        self.train_features = None
        self.train_labels = None

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        data.columns = [col.strip() for col in data.columns]
        return data

    def fit(self, train_data, feature_columns):
        self.train_features = self.scaler.fit_transform(train_data[feature_columns])
        self.train_labels = train_data['genre'].values

    def predict(self, test_data, feature_columns):
        test_features = self.scaler.transform(test_data[feature_columns])
        predictions = []
        for index, test_row in enumerate(test_features):
            neighbors = self._get_neighbors(test_row)
            test_genre = test_data.iloc[index]['genre']
            if self.weighted:
                prediction = self._weighted_vote(neighbors, test_genre)
            else:
                prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)
        return predictions


    def _get_neighbors(self, test_row):
        distances = []
        for i in range(len(self.train_features)):
            dist = self._euclidean_distance(test_row, self.train_features[i])
            distances.append((self.train_labels[i], dist))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]
        return neighbors

    def _weighted_vote(self, neighbors, test_genre):
        class_weights = {}
        for (label, dist) in neighbors:
            weight = 1 / (dist + 1e-5)
            if label in class_weights:
                class_weights[label] += weight
            else:
                class_weights[label] = weight
        predicted_class = max(class_weights, key=class_weights.get)
        sorted_class_weights = dict(sorted(class_weights.items(), key=lambda item: item[1], reverse=True))
        print(f"Test genre: {test_genre}, Predicted: {predicted_class}, Sorted Class Weights: {sorted_class_weights}")
        return predicted_class


    def _euclidean_distance(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def calculate_accuracy(self, test_data, predictions):
        correct = sum(1 for i in range(len(test_data)) if test_data.iloc[i]['genre'] == predictions[i])
        return correct / float(len(test_data)) * 100.0

    def get_confusion_matrix(self, test_data, predictions):
        actual_labels = test_data['genre'].values
        cm = confusion_matrix(actual_labels, predictions, labels=np.unique(actual_labels))
        return cm

if __name__ == "__main__":
    feature_columns = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
    classifier = KNNClassifier(k=5, weighted=True)  # Enable weighted voting
    train_data = classifier.load_data('GenreClassData_30s.csv')
    test_data = classifier.load_data('GenreClassData_30s.csv')
    #classifier.fit(train_data, feature_columns)
    predictions = classifier.predict(test_data, feature_columns)
    accuracy = classifier.calculate_accuracy(test_data, predictions)
    cm = classifier.get_confusion_matrix(test_data, predictions)
    print(f'Accuracy: {accuracy:.2f}%')
    print("Confusion Matrix:\n", cm)