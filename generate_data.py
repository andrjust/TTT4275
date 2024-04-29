import pandas as pd
from sklearn.model_selection import train_test_split

n_train = 80
n_validation = 10
data_set = pd.read_csv('Data/GenreClassData_30s.txt', sep='\t')

train_validation_sets = []
train_sets = []
test_sets = []
val_sets = []

genre_groups = data_set.groupby('GenreID')

for genre, group_data in genre_groups:
    # Split the data for the current genre into training, validation, and test sets
    train_data, test_and_val_data = train_test_split(group_data, test_size=30, random_state=42)
    val_data, test_data = train_test_split(test_and_val_data, test_size=15, random_state=42)
    
    # Append the splits for the current genre to the respective lists
    train_sets.append(train_data)
    val_sets.append(val_data)
    test_sets.append(test_data)

# Concatenate the splits for each genre into the final training, validation, and test sets
train_set = pd.concat(train_sets)
val_set = pd.concat(val_sets)
test_set = pd.concat(test_sets)

train_set.to_csv('Data/train_data_task4.txt', sep='\t', index=False)
test_set.to_csv('Data/test_data_task4.txt', sep='\t', index=False)
val_set.to_csv('Data/val_data_task4.txt', sep='\t', index=False)