import pandas as pd
import torch
import torch_geometric.data as pygd
import torch_geometric.transforms as T


class OsdGraphPyg:

    def __init__(self, path_user_data, path_item_data, path_review_data, path_edges, num_fake_item_feature=-1,
                 num_fake_user_feature=-1, ignore_review_feature=None, ignore_user_feature=None):
        # carico i dati dai file
        self.user_mapping, user_node_id_tensor, feature_user_tensor = self.load_user_from_file(path_user_data)
        self.item_mapping, item_node_id_tensor, feature_item_tensor = self.load_item_from_file(path_item_data)
        self.review_mapping, review_edge_id_tensor, feature_review_tensor, dict_masks, label_review_tensor = self.load_review_from_file(
            path_review_data, ignore_review_feature)

        edge_index = self.load_edges_from_file(path_edges, self.user_mapping, self.item_mapping, self.review_mapping)
        # costruisco il grafo totale, contenente archi di training, test e validation
        self.g = pygd.HeteroData()
        # definisco i nodi
        self.g['user'].num_nodes = len(self.user_mapping)
        self.g['user'].node_id = user_node_id_tensor
        # self.g['user'].x = feature_user_tensor
        self.g['item'].num_nodes = len(self.item_mapping)
        self.g['item'].node_id = item_node_id_tensor
        # non essendoci feature sui nodi, inserisco feature tutte uguali per ogni tipologia di nodo
        if num_fake_user_feature > 0:
            self.g['user'].x = torch.ones(
                (len(self.user_mapping), num_fake_user_feature))  # feature_item_tensor[0].size(0)))
        else:
            self.g['user'].x = feature_user_tensor

        if num_fake_item_feature > 0:
            self.g['item'].x = 0.5 * torch.ones((len(self.item_mapping), num_fake_item_feature))
        else:
            self.g['item'].x = feature_item_tensor
        # definisco gli archi
        self.g['user', 'review', 'item'].edge_index = edge_index.T
        self.g['user', 'review', 'item'].edge_id = review_edge_id_tensor
        self.g['user', 'review', 'item'].edge_attr = feature_review_tensor
        self.g['user', 'review', 'item'].label = label_review_tensor
        self.g['user', 'review', 'item'].train_mask = dict_masks['train_mask']
        self.g['user', 'review', 'item'].val_mask = dict_masks['val_mask']
        self.g['user', 'review', 'item'].test_mask = dict_masks['test_mask']
        trasform = T.ToUndirected()
        self.g = trasform(self.g)

        # costruisco il grafo di training, che è formato dai soli esempi di training
        self.train_g = pygd.HeteroData()
        self.train_g['user'].num_nodes = len(self.user_mapping)
        self.train_g['user'].node_id = user_node_id_tensor
        self.train_g['item'].num_nodes = len(self.item_mapping)
        self.train_g['item'].node_id = item_node_id_tensor
        if num_fake_user_feature>0:
            self.train_g['user'].x = torch.ones(
                (len(self.user_mapping), num_fake_user_feature))  # feature_item_tensor[0].size(0)))
        else:
            self.train_g['user'].x = feature_user_tensor
        if num_fake_item_feature > 0:
            self.train_g['item'].x = 0.5 * torch.ones((len(self.item_mapping), num_fake_item_feature))
        else:
            self.train_g['item'].x = feature_item_tensor
        self.train_g['user', 'review', 'item'].edge_index = edge_index[dict_masks['train_mask']].T
        self.train_g['user', 'review', 'item'].edge_id = review_edge_id_tensor[dict_masks['train_mask']]
        self.train_g['user', 'review', 'item'].edge_attr = feature_review_tensor[dict_masks['train_mask']]
        self.train_g['user', 'review', 'item'].label = label_review_tensor[dict_masks['train_mask']]
        self.train_g['user', 'review', 'item'].train_mask = dict_masks['train_mask'][dict_masks['train_mask']]
        self.train_g = trasform(self.train_g)

        # costruisco il grafo di validation, formato da esempi di training+esempi di validation
        self.val_g = pygd.HeteroData()
        self.val_g['user'].node_id = user_node_id_tensor
        self.val_g['item'].node_id = item_node_id_tensor
        if num_fake_user_feature>0:
            self.val_g['user'].x = torch.ones(
                (len(self.user_mapping), num_fake_user_feature))  # feature_item_tensor[0].size(0)))
        else:
            self.val_g['user'].x = feature_user_tensor
        if num_fake_item_feature > 0:
            self.val_g['item'].x = 0.5 * torch.ones((len(self.item_mapping), num_fake_item_feature))
        else:
            self.val_g['item'].x = feature_item_tensor
        self.val_g['user', 'review', 'item'].edge_index = edge_index[
            torch.logical_or(dict_masks['val_mask'], dict_masks['train_mask'])].T
        self.val_g['user', 'review', 'item'].edge_id = review_edge_id_tensor[
            torch.logical_or(dict_masks['val_mask'], dict_masks['train_mask'])]
        self.val_g['user', 'review', 'item'].edge_attr = feature_review_tensor[
            torch.logical_or(dict_masks['val_mask'], dict_masks['train_mask'])]
        self.val_g['user', 'review', 'item'].label = label_review_tensor[
            torch.logical_or(dict_masks['val_mask'], dict_masks['train_mask'])]
        self.val_g['user', 'review', 'item'].train_mask = dict_masks['train_mask'][
            torch.logical_or(dict_masks['val_mask'], dict_masks['train_mask'])]
        self.val_g['user', 'review', 'item'].val_mask = dict_masks['val_mask'][
            torch.logical_or(dict_masks['val_mask'], dict_masks['train_mask'])]
        self.val_g = trasform(self.val_g)

    def load_user_from_file(self, path_data):
        df_user = pd.read_csv(path_data, index_col='reviewer_id').sort_index()
        mapping = {index: i for i, index in enumerate(df_user.index.unique())}
        node_id_tensor = torch.tensor([uid for uid in range(len(mapping))], dtype=torch.long)
        feature_user_tensor = torch.tensor(df_user.to_numpy(), dtype=torch.float32)
        #print(feature_user_tensor)
        return mapping, node_id_tensor, feature_user_tensor

    def load_item_from_file(self, path_data):
        df_item = pd.read_csv(path_data, index_col='product_id').sort_index()
        mapping = {index: i for i, index in enumerate(df_item.index.unique())}
        node_id_tensor = torch.tensor([uid for uid in range(len(mapping))], dtype=torch.long)
        feature_item_tensor = torch.tensor(df_item.to_numpy(), dtype=torch.float32)
        #print(feature_item_tensor)
        return mapping, node_id_tensor, feature_item_tensor

    def load_review_from_file(self, path_data, ignore_review_feature=None):
        df_review = pd.read_csv(path_data, index_col='review_id')
        if ignore_review_feature is not None:
            for start_name in ignore_review_feature:
                cols_to_remove = [col for col in df_review.columns.to_list() if str(col).startswith(start_name)]
                df_review.drop(columns=cols_to_remove, inplace=True)

        mapping = {index: i for i, index in enumerate(df_review.index.unique())}
        review_id_tensor = torch.tensor([revid for revid in range(len(mapping))], dtype=torch.long)
        # label_tensor = torch.tensor(myutils.one_hot(df_review[['label']].to_numpy(), 2), dtype=torch.float32)
        label_tensor = torch.tensor((df_review['label']).to_numpy(), dtype=torch.float32)
        label_tensor = label_tensor.reshape(label_tensor.size(0), 1)
        mask_dict = {
            'train_mask': torch.tensor(df_review['split'] == 'TRAIN'),
            'val_mask': torch.tensor(df_review['split'] == 'VALIDATION'),
            'test_mask': torch.tensor(df_review['split'] == 'TEST')
        }
        review_feature = torch.tensor(df_review.drop(columns=['split', 'label']).to_numpy(), dtype=torch.float32)
        return mapping, review_id_tensor, review_feature, mask_dict, label_tensor

    def load_edges_from_file(self, path_edges, user_mapping, item_mapping, review_mapping):
        df_assoc = pd.read_csv(path_edges)
        df_assoc['review_id'] = df_assoc['review_id'].apply(lambda x: review_mapping[x])
        df_assoc['product_id'] = df_assoc['product_id'].apply(lambda x: item_mapping[x])
        df_assoc['reviewer_id'] = df_assoc['reviewer_id'].apply(lambda x: user_mapping[x])
        return torch.tensor(df_assoc[['reviewer_id', 'product_id']].to_numpy(), dtype=torch.long)

    def get_graph(self):
        return self.g

    def get_train_graph(self):
        return self.train_g

    def get_val_graph(self):
        return self.val_g

    def get_test_graph(self):
        return self.g

    def get_user_mapping(self):
        return self.user_mapping

    def get_item_mapping(self):
        return self.item_mapping


def main():
    path_assoc = '../data/bingliuhotel/backup/associazioni.csv'
    path_review_data = '../data/bingliuhotel/backup/review_data.csv'
    grafo = OsdGraphPyg(path_assoc, path_assoc, path_review_data, path_assoc)


if __name__ == '__main__':
    main()
