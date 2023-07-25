import torch
import torch.nn as nn
import torch_geometric.nn as pygnn


class ConvolutionLayer(nn.Module):
    def __init__(self, num_node_feature, num_edge_feature, dropout_p):
        super(ConvolutionLayer, self).__init__()

        self.conv1 = pygnn.CGConv(num_node_feature, dim=num_edge_feature, aggr='mean', batch_norm=True)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.conv2 = pygnn.CGConv(num_node_feature, dim=num_edge_feature, aggr='mean', batch_norm=True)
        self.dropout2 = nn.Dropout(p=dropout_p)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.dropout1(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.dropout2(x)
        return x


class Ensembler(nn.Module):
    def __init__(self, num_hidden_feature_user, num_hidden_feature_item, num_hidden_edge_feature,
                 num_out_feature) -> None:
        super(Ensembler, self).__init__()
        self.dense = nn.Linear(num_hidden_feature_item + num_hidden_feature_user,
                               num_out_feature)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x_user, x_item):
        user_part = x_user
        item_part = x_item
        hidden_feature_edge = torch.cat([user_part, item_part], dim=1)
        x = self.dense(hidden_feature_edge)
        return x


class ModelCGmod2(nn.Module):
    def model_name(self):
        return 'GLORIA'

    def __init__(self, data, num_user_feature, num_item_feature, num_edge_feature, dropout_p=0.3):
        super(ModelCGmod2, self).__init__()
        self.dense_edge_1 = nn.Linear(num_edge_feature, 256)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.dense_edge_2 = nn.Linear(256, 128)
        self.dropout_2 = nn.Dropout(p=dropout_p)
        self.dense_edge_3 = nn.Linear(128, 64)

        self.dense_user = nn.Linear(num_user_feature, 16)
        self.dense_item = nn.Linear(num_item_feature, 16)

        self.conv = pygnn.to_hetero(ConvolutionLayer(16, 64, dropout_p=dropout_p), data.metadata(), aggr='mean')

        self.ensembler = Ensembler(16, 16, 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data, edge_index_batch):
        # mod_edge_attr = data['review'].edge_attr
        mod_edge_attr = self.dense_edge_1(data['review'].edge_attr)
        mod_edge_attr = nn.ReLU()(mod_edge_attr)
        mod_edge_attr = self.dropout_1(mod_edge_attr)
        mod_edge_attr = self.dense_edge_2(mod_edge_attr)
        mod_edge_attr = nn.ReLU()(mod_edge_attr)
        mod_edge_attr = self.dropout_2(mod_edge_attr)
        mod_edge_attr = self.dense_edge_3(mod_edge_attr)

        x_dict = data.x_dict
        x_dict['user'] = self.dense_user(x_dict['user'])
        x_dict['item'] = self.dense_item(x_dict['item'])

        x = self.conv(x_dict, data.edge_index_dict, {edge_type: mod_edge_attr for edge_type in data.edge_types})
        x = self.ensembler(x['user'][edge_index_batch.T[0]], x['item'][edge_index_batch.T[1]])
        return self.sigmoid(x)
