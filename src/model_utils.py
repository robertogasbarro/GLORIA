from ModelCGmod2 import ModelCGmod2


def create_model(model_name, graph, dropout_p):
    if model_name == 'ModelCGmod2':
        model = ModelCGmod2(graph, graph['user'].x[0].size(0), graph['item'].x[0].size(0),
                            graph['review'].edge_attr[0].size(0), dropout_p=dropout_p)
    else:
        raise Exception('no model specified')

    return model
