import numpy as np
import pandas as pd
import dgl
import torch
from feature_process import process_user_item_features

# Read training and test data
train_data = pd.read_csv('./food_DR/train.txt', sep=' ')
test_data = pd.read_csv('./food_DR/test_ood.txt', sep=' ')
train_data.columns = ['user_id', 'item_id']
test_data.columns = ['user_id', 'item_id']

# Print some information
print("Maximum item code: {0}".format(train_data['item_id'].max()))

# Add offset
train_data['item_id'] = train_data['item_id'] + train_data['user_id'].nunique()
test_data['item_id'] = test_data['item_id'] + train_data['user_id'].nunique()

# Get the maximum user code and minimum item code in the training set
max_user_id = train_data['user_id'].max()
min_item_id = train_data['item_id'].min()
print("Maximum user code: {0} and minimum item code: {1}".format(max_user_id, min_item_id))

# Get the number of users and items in the training set
num_users = train_data['user_id'].nunique()
num_items = train_data['item_id'].nunique()
max_item_id = train_data['item_id'].max()
print("Number of users: {0} and number of items: {1}".format(num_users, num_items))
print(max_item_id)

# Remove user-item interactions in the test set where the item code exceeds the maximum item code in the training set
test_data = test_data[test_data['item_id'] <= max_item_id]

# Print the size of the filtered test set
print("Size of the filtered test set: ", test_data.shape[0])

# Create graph
print(train_data)
u, v = train_data['user_id'].values, train_data['item_id'].values
print(u)
print(v)
trt_g = dgl.graph((u, v), num_nodes=max_user_id + num_items + 1)
num_nodes = max_user_id + num_items + 1
print(num_nodes)

u, v = test_data['user_id'].values, test_data['item_id'].values
tst_g = dgl.graph((u, v), num_nodes=max_user_id + num_items + 1)

# Process user and item features
user_features, item_features = process_user_item_features('./feature/user_feature.txt',
                                                          './feature/item_feature.txt')
print("Feature shapes:")
print(user_features.shape)
print(item_features.shape)

# Add node features
trt_g.ndata['feat'] = torch.cat([user_features, item_features], dim=0)
tst_g.ndata['feat'] = torch.cat([user_features, item_features], dim=0)

# Print features of some nodes
print("Node features for some users and items:")

# Save the graphs
dgl.save_graphs("./food_causal/food_train_data.bin", trt_g)
dgl.save_graphs("./food_causal/food_test_data.bin", tst_g)