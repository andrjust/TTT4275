import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define the path to your dataset files
train_file_path = 'training_set.csv'
test_file_path = 'testing_set.csv'

# Function to load dataset
def load_dataset(filepath):
    # Assuming data is comma-separated and the first row contains headers
    data = pd.read_csv(filepath)
    return data

# Load training and testing data
train_data = load_dataset(train_file_path)
test_data = load_dataset(test_file_path)

# Extract features and labels
X_train = train_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroids', 'tempo']]
y_train = train_data['genre']
X_test = test_data[['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroids', 'tempo']]
y_test = test_data['genre']

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a kNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the test set results
y_pred = knn.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
