from copy import copy
import networkx as nx
import torch
import torch_geometric.data
from torchviz import make_dot
import os
import matplotlib.pyplot as plt
from typing import Union, Tuple, Dict, Any, Iterable
from src.OsdGraphPyg import OsdGraphPyg


def print_model(model, graph, dl):
    os.environ["PATH"] += os.pathsep + 'D:/Graphviz/bin'
    x = next(iter(dl))
    y = model(graph, x[0])
    print(make_dot(y.mean(), params=dict(model.named_parameters())).render("attached", format="png"))


def inverse_dict(d):
    inverse = {v: k for k, v in d.items()}
    return inverse


def adjust_dict(d):
    keys = [k for k in d.keys()]
    for key in keys:
        value = d[key]
        if isinstance(value, dict):
            internal_d = adjust_dict(value)
            for k in [k for k in internal_d.keys()]:
                internal_d[str(key) + '_' + str(k)] = internal_d.pop(k)
            d.update(internal_d)
            del d[key]
    return d


def heterodata_to_networkx(heterodata: torch_geometric.data.HeteroData, save_feature=False,
                           ignore_node_attr: Union[None, Iterable[str]] = None,
                           ignore_edge_attr: Union[None, Iterable[str]] = None,
                           node_colors: Union[None, Tuple[str, Dict[Any, str]]] = None,
                           edge_colors: Union[None, Tuple[str, Dict[Any, str]]] = None):
    G = nx.Graph()
    node_types = heterodata.node_types
    start_node_index = 0
    node_mapping = {}  # old index : new index
    heterodata_dict = heterodata.to_dict()

    for ntype in node_types:
        old_indices = heterodata[ntype].node_id.numpy()
        new_indices = old_indices + start_node_index
        node_mapping[ntype] = {x: x + start_node_index for x in list(old_indices)}
        attr = heterodata_dict[ntype]
        if not node_colors is None:

            if node_colors[0] in ['type', 'ntype', 'node_type']:
                attr['color'] = [node_colors[1][ntype]] * len(new_indices)
            elif not node_colors[0] in attr.keys():
                raise Exception('chiave specificata non trovata')
            else:
                color_dict = node_colors[1]
                attr_to_check = node_colors[0]
                attr['color'] = [color_dict[int(value.item()) if torch.is_tensor(value) else value] for value in
                                 attr[attr_to_check]]

        if not ignore_node_attr is None:
            for attribute in ignore_node_attr:
                if attribute in attr.keys():
                    del attr[attribute]
        if 'num_nodes' in attr.keys(): del attr['num_nodes']

        new_indices = list(new_indices)
        node_index_with_attr = [(new_indices[i], {k: attr[k][i] for k in attr.keys()}) for i in range(len(new_indices))]

        G.add_nodes_from(node_index_with_attr, node_type=ntype)
        if save_feature:
            raise NotImplementedError()
        start_node_index += len(old_indices)

    G.graph['node_mapping'] = node_mapping
    edge_types = [t for t in heterodata.edge_types if not str(t[1]).startswith('rev_')]

    for etype in edge_types:
        src_ntype = etype[0]
        edge_type = etype[1]
        dst_ntype = etype[2]
        attr = heterodata_dict[etype]

        old_edge_index = heterodata[edge_type].edge_index.T.numpy()
        new_edge_index = list(
            map(lambda x: [node_mapping[src_ntype][x[0]], node_mapping[dst_ntype][x[1]]], old_edge_index))

        if not edge_colors is None:
            if edge_colors[0] in ['type', 'etype', 'edge_type']:
                attr['color'] = [edge_colors[1][edge_type]] * len(new_edge_index)
            elif not edge_colors[0] in attr.keys():
                raise Exception('chiave specificata non trovata')
            else:
                color_dict = edge_colors[1]
                attr_to_check = edge_colors[0]
                attr['color'] = [color_dict[int(value.item()) if torch.is_tensor(value) else value] for value in
                                 attr[attr_to_check]]
        if 'edge_index' in attr.keys(): del attr['edge_index']

        if not ignore_edge_attr is None:
            if isinstance(ignore_edge_attr, list):
                for attribute in ignore_edge_attr:
                    if attribute in attr.keys():
                        del attr[attribute]

        edge_index_with_attr = [(new_edge_index[i][0], new_edge_index[i][1], {k: attr[k][i] for k in attr.keys()})
                                for i in range(len(new_edge_index))]

        G.add_edges_from(edge_index_with_attr, edge_type=edge_type)

    return G


def draw_graph(graph: nx.Graph,
               name_file=None,
               show=True,
               highlighted_node=None,
               title="",
               label_node=True,
               highlighted_node_size: int = 800,
               default_node_size: int = 400,
               first_level_edge_weight: float = 2.0,
               default_level_edge_weight: float = 1.0,
               highlighted_node_color="purple",
               node_type_renaming=None,
               edge_type_renaming=None,
               figure_size=None,
               node_color_dict=None,
               edge_color_dict=None,
               label_node_fontsize=16,
               margin=None):

    if margin is None:
        margin = [0, 0, 1, 1]
    if edge_type_renaming is None:
        edge_type_renaming = {0: 'Non spam review', 1: 'Spam review'}
    if edge_color_dict is None:
        edge_color_dict = {0: 'green', 1: 'red'}
    if node_color_dict is None:
        node_color_dict = {'item': 'blue', 'user': 'orange'}
    if figure_size is None:
        figure_size = [25, 25]
    if node_type_renaming is None:
        node_type_renaming = {'item': 'Product', 'user': 'User'}
    if highlighted_node is not None:
        if highlighted_node not in graph.nodes():
            return
        pos = nx.kamada_kawai_layout(graph, weight=None, scale=2)
        pos[highlighted_node] = (0, 0)  # assegna posizione (0,0) al nodo evidenziato
        node_sizes = [highlighted_node_size if node == highlighted_node else default_node_size for node in
                      graph.nodes()]
        edge_sizes = [first_level_edge_weight if edge[0] == highlighted_node or edge[1] == highlighted_node else default_level_edge_weight for edge in
                      graph.edges()]
        node_colors = [highlighted_node_color if node == highlighted_node else data['color'] for node, data in graph.nodes(data=True)]
        highlighted_node_type = node_type_renaming[graph.nodes[highlighted_node]['node_type']]
    else:
        pos = nx.spring_layout(graph)
        node_sizes = default_node_size
        node_colors = [data['color'] for node, data in graph.nodes(data=True)]
        edge_sizes = [default_level_edge_weight for edge in graph.edges()]
        highlighted_node_type = 'node'

    labels = {node: str(node_type_renaming[data['node_type']])[0] + str(node) for node, data in graph.nodes(data=True)}
    plt.figure(figsize=figure_size)

    nx.draw_networkx(graph, with_labels=label_node, arrows=False,
                     edge_color=[item[2]['color'] for item in list(graph.edges(data=True))],
                     node_color=node_colors,
                     pos=pos,
                     font_size=label_node_fontsize,
                     node_size=node_sizes,
                     labels=labels if label_node else None,
                     width=edge_sizes,
                     )
    #legenda
    legend_node_colors = {'Selected ' + highlighted_node_type: highlighted_node_color}
    legend_node_colors.update({node_type_renaming[k]:v for k,v in node_color_dict.items()})
    legend_edge_colors = {edge_type_renaming[k]:v for k,v in edge_color_dict.items()}

    plt.legend(
        handles=[plt.Line2D([], [], marker='o', linestyle='None', markersize=30, markerfacecolor=color,
                            markeredgecolor=None, label=label)
                 for label, color in legend_node_colors.items()] + [
                    plt.Line2D([], [], marker='_', linestyle='None', markersize=30, markeredgecolor=color,
                               label=label)
                    for label, color in legend_edge_colors.items()], loc='lower right', fontsize=48)

    # plt.title(title, fontdict={'fontsize': 50, 'wrap': True})
    plt.subplots_adjust(bottom=margin[0], left=margin[1], right=margin[2], top=margin[3])
    if not name_file is None:
        plt.savefig(name_file, dpi=250)

    if show:
        plt.show()


def safe_division(a, b, default=0):
    res = a / b if b != 0 else default
    return res


# def add_color_edges(graph: nx.Graph):


if __name__ == '__main__':
    path_user_data = '../data/bingliuhotel_sim_w2v/user_data.csv'
    path_item_data = '../data/bingliuhotel_sim_w2v/item_data.csv'
    path_review_data = '../data/bingliuhotel_sim_w2v/review_data.csv'
    path_edges = '../data/bingliuhotel_sim_w2v/associazioni.csv'
    num_fake_user_feature = 3
    num_fake_item_feature = -1
    ignore_review_feature = []
    ignore_user_feature = ['USER']
    dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                       num_fake_item_feature=num_fake_item_feature, num_fake_user_feature=num_fake_user_feature,
                       ignore_review_feature=ignore_review_feature, ignore_user_feature=ignore_user_feature)

    heterodata = dati.get_test_graph()
    print(heterodata)

    g = heterodata_to_networkx(heterodata, ignore_edge_attr=['train_mask', 'val_mask', 'test_mask', 'edge_attr'],
                               node_colors=('type', {'user': 'blue', 'item': 'yellow'}),
                               edge_colors=('label', {0: 'green', 1: 'red'}))
    draw_graph(g)
