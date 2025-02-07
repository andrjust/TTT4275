from kNNclassifier import KNNClassifier
from optimize_functions import find_optimal_k, find_best_features

potential_features = [
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

#Load data
classifier = KNNClassifier(k=5, weighted=True)
train_data = classifier.load_data('Data/task4/5s_train.txt')
test_data = classifier.load_data('Data/task4/5s_test.txt')
val_data = classifier.load_data('Data/task4/5s_val.txt')

#Find best features:
best_features = find_best_features(train_data, val_data, classifier, potential_features, N=len(potential_features))

#Find optimal k
k = find_optimal_k(train_data, val_data, best_features, (1, 30))

classifier_optimal = KNNClassifier(k, weighted=True)



<<<<<<< HEAD:task3.py
for i in known_features:
    feature_vector = [a for a in known_features if  a!= i]
    features, accuracy = find_best_features(train_data, val_data, classifier, potential_features, requierd_features=feature_vector, N=1)
    print(f'Features: {features}, Accuracy: {accuracy:.2f}%')
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_features = features
=======
classifier_optimal.fit(train_data, best_features)
>>>>>>> ab49641bf8780ed1b357dc073c18384dcf83e5ae:main.py

predictions = classifier.predict(train_data, best_features)
accuracy = classifier.calculate_accuracy(train_data, predictions)
print(f'Train Accuracy: {accuracy:.2f}%')

predictions = classifier.predict(val_data, best_features)
accuracy = classifier.calculate_accuracy(val_data, predictions)
print(f'Val Accuracy: {accuracy:.2f}%')

predictions = classifier.predict(test_data, best_features)
accuracy = classifier.calculate_accuracy(test_data, predictions)
print(f'Test Accuracy: {accuracy:.2f}%')

