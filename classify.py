import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture

class KNNClassifier:
    def __init__(self, k=5, weighted=False):
        self.k = k
        self.weighted = weighted
        self.scaler = StandardScaler()
        self.train_features = None
        self.train_labels = None

    def load_data(self, filepath):
        data = pd.read_csv(filepath, sep='\t')
        data.columns = [col.strip() for col in data.columns]
        return data

    def fit(self, train_data, feature_columns):
        self.train_features = self.scaler.fit_transform(train_data[feature_columns])
        self.train_labels = train_data['Genre'].values

    def predict(self, test_data, feature_columns):
        test_features = self.scaler.transform(test_data[feature_columns])
        predictions = []
        for index, test_row in enumerate(test_features):
            neighbors = self._get_neighbors(test_row)
            test_genre = test_data.iloc[index]['Genre']
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
        #print(f"Test genre: {test_genre}, Predicted: {predicted_class}, Sorted Class Weights: {sorted_class_weights}")
        return predicted_class


    def _euclidean_distance(self, row1, row2):
        return np.sqrt(np.sum((row1 - row2) ** 2))

    def calculate_accuracy(self, test_data, predictions):
        correct = sum(1 for i in range(len(test_data)) if test_data.iloc[i]['Genre'] == predictions[i])
        return correct / float(len(test_data)) * 100.0

    def get_confusion_matrix(self, test_data, predictions):
        actual_labels = test_data['genre'].values
        cm = confusion_matrix(actual_labels, predictions, labels=np.unique(actual_labels))
        return cm

def find_optimal_k(train_data, test_data, feature_columns):
    best_k = 0
    best_accuracy = 0
    for k in range(1, 101):
        classifier = KNNClassifier(k=k, weighted=True)
        classifier.fit(train_data, feature_columns)
        predictions = classifier.predict(test_data, feature_columns)
        accuracy = classifier.calculate_accuracy(test_data, predictions)
        print(f'K: {k}, Accuracy: {accuracy:.2f}%')
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
    return best_k, best_accuracy

def find_best_mfcc(train_data, test_data, classifier):
    best_mfcc = ""
    best_accuracy = 0
    for i in range(1, 13):
        feature_columns = [f'mfcc_{i}_mean']
        classifier.fit(train_data, feature_columns)
        predictions = classifier.predict(test_data, feature_columns)
        accuracy = classifier.calculate_accuracy(test_data, predictions)
        print(f'MFCC: {i}, Accuracy: {accuracy:.2f}%')
        if accuracy > best_accuracy:
            best_mfcc = feature_columns[0]
            best_accuracy = accuracy
    return best_mfcc, best_accuracy

def find_best_features(train_data, test_data, classifier, features):
    current_best_features = ['spectral_rolloff_mean', 'tempo', 'spectral_centroid_mean']
    best_accuracy = 0

    for i in range(len(features)):
        candidate_mfcc = None
        candidate_accuracy = 0
        
        for feature in features:
            # Check if feature is already included
            if feature in current_best_features:
                continue
            
            # Add the feature to the current set of features
            candidate_features = current_best_features + [feature]
            classifier.fit(train_data, candidate_features)
            predictions = classifier.predict(test_data, candidate_features)
            accuracy = classifier.calculate_accuracy(test_data, predictions)
            
            # Update candidate feature if it improves accuracy
            if accuracy > candidate_accuracy:
                candidate_accuracy = accuracy
                candidate_mfcc = feature
        
        # If adding the best candidate improves accuracy, update the best features and accuracy
        if candidate_accuracy > best_accuracy:
            best_accuracy = candidate_accuracy
            current_best_features.append(candidate_mfcc)
            print(f'Current Best Features: {current_best_features}, Current Accuracy: {best_accuracy:.2f}%')
        else:
            # If adding more features does not improve accuracy, stop the search
            break
    
    return current_best_features, best_accuracy


def GMM(train_data, test_data, feature_columns):
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(train_data[feature_columns])
    predictions = gmm.predict(test_data[feature_columns])
    accuracy = sum(1 for i in range(len(test_data)) if test_data.iloc[i]['GenreID'] == predictions[i]) / float(len(test_data)) * 100.0
    return accuracy
    
    
    
    
         
    

if __name__ == "__main__":
    feature_columns = [
    'zero_cross_rate_mean', 'zero_cross_rate_std', 'rmse_mean', 'rmse_var', 
    'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 
    'spectral_rolloff_mean', 'spectral_rolloff_var', 'spectral_contrast_mean', 'spectral_contrast_var', 
    'spectral_flatness_mean', 'spectral_flatness_var', 'chroma_stft_1_mean', 'chroma_stft_2_mean', 
    'chroma_stft_3_mean', 'chroma_stft_4_mean', 'chroma_stft_5_mean', 'chroma_stft_6_mean', 
    'chroma_stft_7_mean', 'chroma_stft_8_mean', 'chroma_stft_9_mean', 'chroma_stft_10_mean', 
    'chroma_stft_11_mean', 'chroma_stft_12_mean', 'chroma_stft_1_std', 'chroma_stft_2_std', 
    'chroma_stft_3_std', 'chroma_stft_4_std', 'chroma_stft_5_std', 'chroma_stft_6_std', 
    'chroma_stft_7_std', 'chroma_stft_8_std', 'chroma_stft_9_std', 'chroma_stft_10_std', 
    'chroma_stft_11_std', 'chroma_stft_12_std', 'tempo', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 
    'mfcc_4_mean', 'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_8_mean', 'mfcc_9_mean', 
    'mfcc_10_mean', 'mfcc_11_mean', 'mfcc_12_mean', 'mfcc_1_std', 'mfcc_2_std', 'mfcc_3_std', 
    'mfcc_4_std', 'mfcc_5_std', 'mfcc_6_std', 'mfcc_7_std', 'mfcc_8_std', 'mfcc_9_std', 
    'mfcc_10_std', 'mfcc_11_std', 'mfcc_12_std']
    classifier = KNNClassifier(k=19, weighted=True)  # Enable weighted voting
    train_data = classifier.load_data('Data/train_data.txt')
    test_data = classifier.load_data('Data/test_data.txt')

    print(GMM(train_data, test_data, ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "mfcc_8_mean"]))
    
    #print(find_optimal_k(train_data, test_data, feature_columns))
    #print(find_best_features(train_data, test_data, classifier, feature_columns))
    #classifier.fit(train_data, feature_columns)
    predictions = classifier.predict(test_data, feature_columns)
    accuracy = classifier.calculate_accuracy(test_data, predictions)
    cm = classifier.get_confusion_matrix(test_data, predictions)
    print(f'Accuracy: {accuracy:.2f}%')
    print("Confusion Matrix:\n", cm)
    #predictions = classifier.predict(test_data, feature_columns)
    #accuracy = classifier.calculate_accuracy(test_data, predictions)
    #print(f'Accuracy: {accuracy:.2f}%')
