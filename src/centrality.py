import configparser
import itertools
import os

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from networkx import NetworkXNoPath
from tqdm import tqdm

from OsdGraphPyg import OsdGraphPyg
from utils import heterodata_to_networkx, draw_graph, inverse_dict, safe_division


def main():
    parser = configparser.ConfigParser()
    parser.read('config.conf')
    config = parser['centrality']

    path_user_data = config.get('path_user_data')
    path_item_data = config.get('path_item_data')
    path_review_data = config.get('path_review_data')
    path_edges = config.get('path_edges')
    num_fake_item_feature = config.getint('num_fake_item_feature')
    num_fake_user_feature = config.getint('num_fake_user_feature')
    ignore_review_feature = eval(config.get('ignore_review_feature'))
    ignore_user_feature = eval(config.get('ignore_user_feature'))
    folder_name = config.get('folder_name')

    dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                       num_fake_item_feature=num_fake_item_feature, num_fake_user_feature=num_fake_user_feature,
                       ignore_review_feature=ignore_review_feature, ignore_user_feature=ignore_user_feature)

    graph = dati.get_test_graph()
    print(graph)

    print('Building networkx graph')
    graph_nx = heterodata_to_networkx(heterodata=graph,
                                      ignore_edge_attr=['train_mask', 'val_mask', 'test_mask', 'edge_attr'],
                                      node_colors=('type', {'user': 'orange', 'item': 'blue'}),
                                      edge_colors=('label', {0: 'green', 1: 'red'}))

    compute_centralities(dati, graph_nx, folder_name)


def compute_centralities(dati, graph_nx, folder_name):
    graph = dati.get_test_graph()
    edges = np.array([graph['review'].edge_index[0].numpy(),
                      graph['review'].edge_index[1].numpy(),
                      graph['review'].label.numpy().reshape(-1)]).T
    node_mapping_pyg_nx = graph_nx.graph['node_mapping']

    pyg_item_mapping = inverse_dict(dati.get_item_mapping())
    pyg_user_mapping = inverse_dict(dati.get_user_mapping())

    out_spam, out_non_spam, in_spam, in_non_spam = compute_inout_edges(edges)
    metrics = {'user':
                   {'id_user': pyg_user_mapping,
                    'id_node_nx': node_mapping_pyg_nx['user'],
                    'spam': out_spam,
                    'non_spam': out_non_spam},
               'item':
                   {'id_item': pyg_item_mapping,
                    'id_node_nx': node_mapping_pyg_nx['item'],
                    'spam': in_spam,
                    'non_spam': in_non_spam}
               }
    print('Computing in/out spam/non-spam')
    spam_edges = [(u, v) for u, v, e in graph_nx.edges(data=True) if e['label'] == 1]
    non_spam_edges = [(u, v) for u, v, e in graph_nx.edges(data=True) if e['label'] == 0]
    subgraph_spam = graph_nx.edge_subgraph(spam_edges)
    subgraph_non_spam = graph_nx.edge_subgraph(non_spam_edges)

    # draw_graph(graph_nx)
    print("Computing degree centrality")
    degree_centrality = nx.degree_centrality(graph_nx)
    save_metrics(metrics, 'degree_centrality', degree_centrality, node_mapping_pyg_nx)
    print("Computing degree centrality on spam subgraph")
    subgraph_spam_degree_centrality = nx.degree_centrality(subgraph_spam)
    save_metrics(metrics, 'subgraph_spam_degree_centrality', subgraph_spam_degree_centrality, node_mapping_pyg_nx)
    print("Computing degree centrality on non-spam subgraph")
    subgraph_non_spam_degree_centrality = nx.degree_centrality(subgraph_non_spam)
    save_metrics(metrics, 'subgraph_non_spam_degree_centrality', subgraph_non_spam_degree_centrality,
                 node_mapping_pyg_nx)

    # BETWEENNESS

    print("Computing betweenness centrality")
    betweenness_centrality = nx.betweenness_centrality(graph_nx, normalized=True, endpoints=True)
    save_metrics(metrics, 'betweenness_centrality', betweenness_centrality, node_mapping_pyg_nx)

    print("Computing betweenness centrality on spam subgraph")
    betweenness_centrality_spam = nx.betweenness_centrality(subgraph_spam, normalized=True, endpoints=True)
    save_metrics(metrics, 'betweenness_centrality_spam', betweenness_centrality_spam, node_mapping_pyg_nx)

    print("Computing betweenness centrality on non-spam subgraph")
    betweenness_centrality_non_spam = nx.betweenness_centrality(subgraph_non_spam, normalized=True, endpoints=True)
    save_metrics(metrics, 'betweenness_centrality_non_spam', betweenness_centrality_non_spam, node_mapping_pyg_nx)

    df_metrics_user = pd.DataFrame(metrics['user'])
    df_metrics_item = pd.DataFrame(metrics['item'])

    df_metrics_user['spam_ratio'] = df_metrics_user['spam'] / (df_metrics_user['spam'] + df_metrics_user['non_spam'])
    df_metrics_item['spam_ratio'] = df_metrics_item['spam'] / (df_metrics_item['spam'] + df_metrics_item['non_spam'])

    df_metrics_item.fillna(0, inplace=True)
    df_metrics_user.fillna(0, inplace=True)

    current_path = os.path.abspath(__file__)
    project_path = os.path.abspath(os.path.join(current_path, "../../"))
    path_experiment = project_path + '/explain/' + folder_name + '/'
    os.makedirs(path_experiment, exist_ok=True)

    cp = os.getcwd()
    os.chdir(path_experiment)

    df_metrics_user.to_csv('metriche_user.csv', index_label='pyg_node_id')
    df_metrics_item.to_csv('metriche_item.csv', index_label='pyg_node_id')

    draw_graph(graph_nx, 'grafo.svg', False)
    draw_graph(subgraph_spam, 'grafo_spam.svg', False)
    draw_graph(subgraph_non_spam, 'grafo_non_spam.svg', False)


def split_metrics(metrics, node_mapping):
    # vecchio:nuovo -> nuovo:vecchio
    user_mapping = {v: k for k, v in node_mapping['user'].items()}
    item_mapping = {v: k for k, v in node_mapping['item'].items()}

    user_dict = {user_mapping[k]: v for k, v in metrics.items() if k in user_mapping.keys()}
    item_dict = {item_mapping[k]: v for k, v in metrics.items() if k in item_mapping.keys()}

    return {'user': user_dict, 'item': item_dict}


def compute_inout_edges(edges):
    out_spam = {id_utente: 0 for id_utente in np.unique(edges.T[0])}
    out_non_spam = {id_utente: 0 for id_utente in np.unique(edges.T[0])}
    in_spam = {id_item: 0 for id_item in np.unique(edges.T[1])}
    in_non_spam = {id_item: 0 for id_item in np.unique(edges.T[1])}

    for edge in tqdm(edges):
        out_spam[edge[0]] += 1 * (edge[2] == 1)
        out_non_spam[edge[0]] += 1 * (edge[2] == 0)
        in_spam[edge[1]] += 1 * (edge[2] == 1)
        in_non_spam[edge[1]] += 1 * (edge[2] == 0)

    return out_spam, out_non_spam, in_spam, in_non_spam


def save_metrics(metrics_list, metric_name, metric_dict, node_mapping):
    m = split_metrics(metric_dict, node_mapping)
    metrics_list['user'][metric_name] = m['user']
    metrics_list['item'][metric_name] = m['item']


def compute_all_shortest_path(G):
    # carico tutti i nodi
    nodes = list(G.nodes)
    edges = list(G.edges)
    # costruisco tutte le coppie di nodi
    print('couple of nodes')
    to_do = {(x, y) for (x, y) in itertools.product(nodes, nodes) if x < y}
    # couples_of_nodes = {(x, y) for (x, y) in couples_of_nodes if x < y}

    # to_do = couples_of_nodes.copy()
    shortest_paths = {}

    next_nodes = {x: {x} for x in nodes}

    for edge in tqdm(edges):
        edge = (edge[1], edge[0]) if edge[0] >= edge[1] else edge
        shortest_paths[edge] = [list(edge)]
        to_do.remove(edge)
        next_nodes[edge[0]].add(edge[1])
        next_nodes[edge[1]].add(edge[0])

    while len() > 0:
        print(str(len(to_do)), 'left')

        src_to_consider = {x for (x, y) in to_do}
        dst_to_consider = {y for (x, y) in to_do}
        new_couples = set()
        actual_sp = [(e, p) for e, p in shortest_paths.items() if e[0] in src_to_consider]
        for (endpoints, paths) in actual_sp:
            new_targets = next_nodes[endpoints[1]]
            for new_target in new_targets & dst_to_consider:
                new_endpoints = (endpoints[0], new_target)
                if new_endpoints in to_do:
                    if not new_endpoints in shortest_paths.keys():
                        shortest_paths[new_endpoints] = []
                    for path in paths:
                        new_path = path[:]
                        new_path.append(new_target)
                        shortest_paths[new_endpoints].append(new_path)
                    new_couples.add(new_endpoints)
        if len(new_couples) == 0:
            print('No more path')
            return shortest_paths
        to_do.difference_update(new_couples)
    print('Finish')
    return shortest_paths


def hetero_betweennes_centrality(G, node_types, normalization=False):
    """
    Funzione che calcola la betweennes centrality in base al tipo di nodo - troppo complessa
    :param G:
    :param node_types:
    :param normalization:
    :return:
    """
    centralities = {}
    nodes = {t: [n for n, data in G.nodes(data=True) if data['node_type'] == t] for t in node_types}
    other_nodes_dict = {t: [n for n, data in G.nodes(data=True) if data['node_type'] != t] for t in node_types}
    for t in node_types:
        print('Type node ', t)
        t_nodes = nodes[t]
        other_nodes = other_nodes_dict[t]
        other_couples = [(x, y) for (x, y) in list(itertools.product(other_nodes, other_nodes)) if x < y]
        other_shortest_paths = {}
        print('Computing shortest paths')
        """for couple in tqdm(other_couples):
            try:
                all_shortest_paths = [p for p in nx.all_shortest_paths(G, source=couple[0], target=couple[1],
                                                                       method='bellman-ford')]
                other_shortest_paths[couple] = all_shortest_paths
            except NetworkXNoPath as np:
                other_shortest_paths[couple] = []"""
        print('Total task:', len(other_couples))
        with parallel_backend('threading', n_jobs=10):
            result = Parallel(n_jobs=10, verbose=0)(
                delayed(job_compute_path)(G, couple) for couple in tqdm(other_couples))
        for d in result:
            other_shortest_paths.update(d)

        other_shortest_paths.update({(y, x): l for (x, y), l in other_shortest_paths.items()})
        print('Computing centralities')
        for n in tqdm(t_nodes):
            N = len(other_couples)
            s = 0
            for couple in other_shortest_paths.keys():
                sigma_sd = len(other_shortest_paths[couple])
                sigma_sd_i = len([p for p in other_shortest_paths[couple] if n in p])
                s += safe_division(sigma_sd_i, sigma_sd)
            if normalization:
                s = s / ((N - 1) * (N - 2))
            centralities[n] = s
    return centralities


def job_compute_path(G, couple):
    try:
        # print(couple)
        all_shortest_paths = [p for p in nx.all_shortest_paths(G, source=couple[0], target=couple[1],
                                                               method='bellman-ford')]
        return {couple: all_shortest_paths}
    except NetworkXNoPath as np:
        return {couple: []}


if __name__ == '__main__':
    main()
