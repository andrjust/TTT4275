import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data into a DataFrame
df = pd.read_csv('Data/GenreClassData_30s.txt', sep='\t')

# Group data by genre
grouped = df.groupby('Genre')

# Plot feature distributions for each class
features = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']

for feature in features:
    plt.figure(figsize=(10, 6))
    for genre, group in grouped:
        if genre in ['pop', 'metal', 'disco', 'classical']:
            sns.histplot(group[feature], kde=True, label=genre, alpha=0.5)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.show()