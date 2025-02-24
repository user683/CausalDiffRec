import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder


def process_user_item_features(user_feat_path, item_feat_path, target_dim=64):  # Default is 64
    # Read user feature data
    user_feat = pd.read_csv(user_feat_path, sep=" ")

    # One-hot encode user IDs
    ohe_user = OneHotEncoder()
    user_id_encoded = ohe_user.fit_transform(user_feat[["user_id"]]).toarray()

    # Convert the encoded user IDs to a DataFrame
    user_id_encoded_df = pd.DataFrame(user_id_encoded,
                                      columns=[f'user_id_{i}' for i in range(user_id_encoded.shape[1])])

    # Merge the encoded user IDs with other user features
    user_features_tensor = torch.Tensor(user_id_encoded_df.values)

    # Read item feature data
    item_feat = pd.read_csv(item_feat_path, sep=' ')

    # One-hot encode item IDs
    ohe_item = OneHotEncoder()
    item_id_encoded = ohe_item.fit_transform(item_feat[["item_id"]]).toarray()

    # Convert the encoded item IDs to a DataFrame
    item_id_encoded_df = pd.DataFrame(item_id_encoded,
                                      columns=[f'item_id_{i}' for i in range(item_id_encoded.shape[1])])

    item_features_tensor = torch.Tensor(item_id_encoded_df.values)

    # Create user feature embedding layer
    user_embedding = nn.Linear(user_features_tensor.shape[1], target_dim)
    # Create item feature embedding layer
    item_embedding = nn.Linear(item_features_tensor.shape[1], target_dim)

    # Apply embedding layers
    user_features_embedded = user_embedding(user_features_tensor)
    item_features_embedded = item_embedding(item_features_tensor)

    # Final features
    final_user_features = torch.relu(user_features_embedded)
    final_item_features = torch.relu(item_features_embedded)

    return final_user_features, final_item_features