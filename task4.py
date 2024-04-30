from classify import KNNClassifier, find_best_features, find_optimal_k

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

#feature_vector = ["spectral_rolloff_mean", "mfcc_1_mean", "spectral_centroid_mean", "tempo"]

#best_features = ['spectral_rolloff_mean', 'spectral_rolloff_var', 'rmse_var', 'chroma_stft_11_std', 'mfcc_9_mean', 'chroma_stft_11_mean', 'spectral_flatness_mean']
classifier = KNNClassifier(k=29, weighted=True)  # Enable weighted voting
train_data = classifier.load_data('Data/task4/10s_train.txt')
test_data = classifier.load_data('Data/task4/10s_test.txt')
val_data = classifier.load_data('Data/task4/10s_val.txt')
best_features = ['rmse_var', 'spectral_flatness_var', 'mfcc_4_mean', 'spectral_rolloff_mean', 'zero_cross_rate_mean', 'rmse_mean', 'mfcc_6_mean', 'spectral_centroid_mean']

#Find the best k
#best_k, accuracy = find_optimal_k(train_data, val_data, best_features, (1, 30))
#best_k = int(best_k)
#print(f'Best K: {best_k}, Accuracy: {accuracy:.2f}%')
#classifier = KNNClassifier(k=best_k, weighted=True)  # Enable weighted voting

#Find the best features
#best_features, accuracy = find_best_features(train_data, val_data, classifier, potential_features, N=len(potential_features))
#print(f'Best Features: {best_features}, Accuracy: {accuracy:.2f}%')


classifier.fit(train_data, best_features)
predictions = classifier.predict(val_data, best_features)
accuracy = classifier.calculate_accuracy(val_data, predictions)
print(f'Val Accuracy: {accuracy:.2f}%')


classifier.fit(train_data, best_features)
predictions = classifier.predict(test_data, best_features)
accuracy = classifier.calculate_accuracy(test_data, predictions)
print(f'Test Accuracy: {accuracy:.2f}%')

classifier.fit(train_data, best_features)
predictions = classifier.predict(train_data, best_features)
accuracy = classifier.calculate_accuracy(train_data, predictions)
print(f'Train Accuracy: {accuracy:.2f}%')
