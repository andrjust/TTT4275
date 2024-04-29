from classify import KNNClassifier

if __name__ == "__main__":
    feature_vector = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
    classifier = KNNClassifier(k=5, weighted=True)  # Enable weighted voting
    train_data = classifier.load_data('Data/train_data_task1.txt')
    test_data = classifier.load_data('Data/test_data_task1.txt')
    classifier.fit(train_data, feature_vector)
    
    predictions = classifier.predict(test_data, feature_vector)
    test_accuracy = classifier.calculate_accuracy(test_data, predictions)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    predictions = classifier.predict(train_data, feature_vector)
    train_accuracy = classifier.calculate_accuracy(train_data, predictions)
    print(f'Train Accuracy: {train_accuracy:.2f}%')
    