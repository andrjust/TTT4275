import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np

n_train = 80
n_validation = 10
data = pd.read_csv('Data/GenreClassData_10s.txt', delimiter='\t')
random.seed(42)

track_ids = data['TrackID'].tolist()
genre_ids = data['GenreID'].tolist()
genre_track_dict = {key: value for key, value in zip(track_ids, genre_ids)}
genre_track_list = list(genre_track_dict.items())

genre_id_list = [[] for _ in range(10)]

for key, value in genre_track_list:
    genre_id_list[value].append(key)

test_list = []
val_list= []
train_list = []
 
for list in genre_id_list:
    random.shuffle(list)
    lengths = [int(len(list) * p) for p in [0.15,0.15, 0.70]]
    parts = [list[:lengths[0]], list[lengths[0]:sum(lengths[:2])], list[sum(lengths[:2]):]]
    test_list.extend(parts[0])
    val_list.extend(parts[1])
    train_list.extend(parts[2])
    
            
train_data = data[data['TrackID'].isin(train_list)]
val_data = data[data['TrackID'].isin(val_list)]
test_data = data[data['TrackID'].isin(test_list)]

train_data.to_csv('Data/task4/10s_train.txt', sep='\t', index=False)
test_data.to_csv('Data/task4/10s_test.txt', sep='\t', index=False)
val_data.to_csv('Data/task4/10s_val.txt', sep='\t', index=False)



     
            
        

#train_data.to_csv('Data/task4/5s_train.txt', sep='\t', index=False)
#test_data.to_csv('Data/task4/5s_test.txt', sep='\t', index=False)
#val_data.to_csv('Data/task4/5s_val.txt', sep='\t', index=False)