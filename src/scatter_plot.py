import configparser

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from sklearn.cluster import KMeans


def main():
    parser = configparser.ConfigParser()
    parser.read('config.conf')
    config = parser['scatter_plot']

    path_metrics = config.get('path_metrics')
    dataset_name = config.get('dataset_name')
    save_folder = config.get('save_folder')
    node_type = config.get('node_type')
    k_anomalie = config.getint('k_anomalie')
    nodes_to_label = eval(config.get('nodes_to_label'))
    x_name = config.get('x_name')
    y_name = config.get('y_name')
    id_node_label = config.get('id_node_label')
    x_axis_name = config.get('x_axis_name')
    y_axis_name = config.get('y_axis_name')
    title = config.get('title')


    """    path_metrics = '../explain/hotel_def/metriche_user.csv'
        dataset_name = 'hotel'
        save_folder = '../plot'
        node_type = 'user'
        k_anomalie = 1
        nodes_to_label = []
        x_name = 'spam'
        y_name = 'betweenness_centrality'
        id_node_label = 'id_node_nx'
        x_axis_name = '#spam'
        y_axis_name = 'centrality'
        title = ''
    """
    save_folder = save_folder + '/' + dataset_name + '/'
    os.makedirs(save_folder, exist_ok=True)

    df_metrics = pd.read_csv(path_metrics)
    print(df_metrics.columns)

    scatter_plot(df_metrics,
                 x_name,
                 y_name,
                 node_type,
                 dataset_name,
                 save_folder,
                 id_node_label=id_node_label,
                 k_anomalie=k_anomalie,
                 x_axis_name=x_axis_name,
                 y_axis_name=y_axis_name,
                 nodes_to_label=nodes_to_label,
                 title=title
                 )


def scatter_plot(df,
                 x_name,
                 y_name,
                 node_type,
                 title_info,
                 save_folder,
                 id_node_label='id_node_nx',
                 k_anomalie=1,
                 title=None,
                 x_axis_name=None,
                 y_axis_name=None,
                 nodes_to_label=None,
                 title_fontsize=24,
                 axis_ticks_fontsize=18,
                 axis_label_fontsize=28,
                 node_label_fontsize=16,
                 ):
    fig = plt.figure(figsize=[8, 6])

    plt.scatter(df[x_name], df[y_name], s=25, c='red')

    # invece di stampare gli id di tutti i nodi, stampo solo i k più lontani dal centroide, che dovrebbero essere punti più anomali.
    if k_anomalie > 0:
        kmeans = KMeans(n_clusters=1).fit(df[[x_name, y_name]])
        centroid = kmeans.cluster_centers_[0]
        distances = np.sqrt(((df[[x_name, y_name]] - centroid) ** 2).sum(axis=1))
        sorted_indices = distances.argsort()[::-1]
        k_furthest_points = df.iloc[sorted_indices[:k_anomalie]]

        print(k_furthest_points[[id_node_label, x_name, y_name]])
        for p in k_furthest_points.iterrows():
            id = p[1][id_node_label]
            x = p[1][x_name]
            y = p[1][y_name]
            plt.annotate(str(int(id)), (x, y), fontsize=node_label_fontsize)
    if isinstance(nodes_to_label, list):
        for n in nodes_to_label:
            record = df[df[id_node_label] == n].to_dict('records')[0]
            p = (record[x_name], record[y_name])
            plt.annotate(str(int(n)), p, fontsize=node_label_fontsize)

    plt.xlabel(x_axis_name if x_axis_name is not None else to_title(x_name), fontdict={'fontsize': axis_label_fontsize})
    plt.ylabel(y_axis_name if y_axis_name is not None else to_title(y_name), fontdict={'fontsize': axis_label_fontsize})
    plt.xticks(fontsize=axis_ticks_fontsize)
    plt.yticks(fontsize=axis_ticks_fontsize)
    if title is not None:
        t = title if title is not None else f"Scatter plot {to_title(y_name)} vs {to_title(x_name)} for {node_type} nodes ({title_info})"
        if len(t) > 0:
            plt.title(t, fontdict={'fontsize': title_fontsize, 'wrap': True, 'color': 'black', 'weight': 'bold'})
        else:
            plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.show()
    t = f"Scatter plot {to_title(y_name)} vs {to_title(x_name)} for {node_type} nodes ({title_info})" if len(
        t) == 0 else t
    fig.savefig(save_folder + t + '.png', format='png')


def to_title(s):
    words = str(s).split(" ")
    fin_text = ""
    for w in words:
        if '_' in w:
            words2 = w.split('_')
            for w2 in words2:
                fin_text += ' ' + w2.capitalize()
        else:
            fin_text += ' ' + w
    return fin_text


if __name__ == '__main__':
    main()
