import pandas as pd
import seaborn as sns

# Load the data into a DataFrame
df = pd.read_csv('Data/GenreClassData_30s.txt', sep=',')

# Define the features and genres of interest
features_of_interest = ['spectral_rolloff_mean', 'mfcc_1_mean', 'spectral_centroid_mean', 'tempo']
genres_of_interest = ['pop', 'metal', 'disco', 'classical']

# Filter the DataFrame for genres of interest
filtered_df = df[df['Genre'].isin(genres_of_interest)]

# Group data by genre
grouped = filtered_df.groupby('Genre')

# Plot histograms for each feature
for feature in features_of_interest:
    sns.displot(data=filtered_df, x=feature, hue='Genre', kind='hist', kde=True, alpha=0.5)