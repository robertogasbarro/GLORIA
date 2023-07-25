import configparser
import os
from OsdGraphPyg import OsdGraphPyg
import networkx as nx
from utils import heterodata_to_networkx, draw_graph, inverse_dict, safe_division

def main():

    parser = configparser.ConfigParser()
    parser.read('config.conf')
    config = parser['graph_plot']

    path_user_data = config.get('path_user_data')
    path_item_data = config.get('path_item_data')
    path_review_data = config.get('path_review_data')
    path_edges = config.get('path_edges')
    folder_name = config.get('folder_name')
    node_to_print = eval(config.get('node_to_print'))
    radius = config.getint('radius')
    label_nodes = config.getboolean('label_nodes')

    dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                       num_fake_item_feature=1, num_fake_user_feature=1)

    graph = dati.get_test_graph()
    print(graph)
    print('Building networkx graph')
    graph_nx = heterodata_to_networkx(heterodata=graph,
                                      ignore_edge_attr=['train_mask', 'val_mask', 'test_mask', 'edge_attr'],
                                      node_colors=('type', {'user': 'orange', 'item': 'blue'}),
                                      edge_colors=('label', {0: 'green', 1: 'red'}))

    spam_edges = [(u, v) for u, v, e in graph_nx.edges(data=True) if e['label'] == 1]
    non_spam_edges = [(u, v) for u, v, e in graph_nx.edges(data=True) if e['label'] == 0]
    subgraph_spam = graph_nx.edge_subgraph(spam_edges)
    subgraph_non_spam = graph_nx.edge_subgraph(non_spam_edges)

    print('Plotting...')
    for node in node_to_print:
        print_subgraph(graph_nx, node, k=radius, folder_name=folder_name, label_node=label_nodes)
        print_subgraph(subgraph_spam, node, k=radius, folder_name=folder_name, graph_name='Subgraph spam', label_node=label_nodes)
        print_subgraph(subgraph_non_spam, node, k=radius, folder_name=folder_name, graph_name='Subgraph non spam',
                       label_node=label_nodes)


def print_subgraph(graph_nx, node, k=2, folder_name=None, graph_name=None, label_node=True):
    os.makedirs('../plot/' + folder_name, exist_ok=True)
    if node not in graph_nx.nodes():
        return
    ego_graph = nx.ego_graph(graph_nx, node, radius=k)
    type = graph_nx.nodes(data=True)[node]['node_type']
    title = f"Neighborhood for node {'product' if type == 'item' else 'reviewer'} {node} of radius {k} {'(' + graph_name + ')' if graph_name is not None else ''}"
    draw_graph(ego_graph, highlighted_node=node, title=title, name_file='../plot/' + folder_name + '/' + title + '.png',
               label_node=label_node)


if __name__ == '__main__':
    main()
