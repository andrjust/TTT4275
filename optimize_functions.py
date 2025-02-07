from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from kNNclassifier import KNNClassifier

def find_optimal_k(train_data, test_data, feature_columns, interval=(1, 30)):
    best_k = 0
    best_accuracy = 0
    k_list = []
    accuracy_list = []
    for k in range(interval[0], interval[1]):
        classifier = KNNClassifier(k=k, weighted=True)
        classifier.fit(train_data, feature_columns)
        predictions = classifier.predict(test_data, feature_columns)
        accuracy = classifier.calculate_accuracy(test_data, predictions)
        print(f'K: {k}, Accuracy: {accuracy:.2f}%')
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
        k_list.append(k)
        accuracy_list.append(accuracy)
    plt.plot(k_list, accuracy_list)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('K vs Accuracy')
    plt.show()
    return best_k, best_accuracy


def find_best_features(train_data, test_data, classifier, potential_features, requierd_features=[], N = 1):
    current_best_features = requierd_features
    best_accuracy = 0

    for i in range(N):
        candidate_mfcc = None
        candidate_accuracy = 0
        for feature in potential_features:
            if feature in current_best_features:
                continue
            
            candidate_features = current_best_features + [feature]
            classifier.fit(train_data, candidate_features)
            predictions = classifier.predict(test_data, candidate_features)
            accuracy = classifier.calculate_accuracy(test_data, predictions)
            
            if accuracy > candidate_accuracy:
                candidate_accuracy = accuracy
                candidate_mfcc = feature
        
        if candidate_accuracy > best_accuracy:
            best_accuracy = candidate_accuracy
            current_best_features.append(candidate_mfcc)
            print(f'Current Best Features: {current_best_features}, Current Accuracy: {best_accuracy:.2f}%')
        else:
            break

    return current_best_features, best_accuracy


def GMM(train_data, test_data, feature_columns):
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(train_data[feature_columns])
    predictions = gmm.predict(test_data[feature_columns])
    accuracy = sum(1 for i in range(len(test_data)) if test_data.iloc[i]['GenreID'] == predictions[i]) / float(len(test_data)) * 100.0
    return accuracy