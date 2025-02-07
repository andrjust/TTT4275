from classify import KNNClassifier, find_best_features, find_optimal_k

classifier = KNNClassifier(k=29, weighted=True)
feature_vector = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]
train_data = classifier.load_data('Data/train_data_task1.txt')
test_data = classifier.load_data('Data/test_data_task1.txt')

find_optimal_k(train_data, test_data, feature_vector, (1, 100))



