import random
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from learning_utils import train
from utils import adjust_dict

global_params = {}


def set_global_param(key, value):
    global_params[key] = value


def set_global_params(param_dict):
    global_params.update(param_dict)


def get_global_param(key):
    return global_params[key]


def initialize_global_params():
    global_params = {}


def start_hyperparameters_optimization(create_model_function, create_dataloader_function,
                                       train_graph, validation_graph, optimizing_metric='val_loss',
                                       **config):
    initialize_global_params()
    set_global_param('train_graph', train_graph)
    set_global_param('create_model_function', create_model_function)
    set_global_param('create_dataloader_function', create_dataloader_function)
    set_global_param('validation_graph', validation_graph)
    set_global_param('optimizing_metric', optimizing_metric)
    max_evals = config['max_iter_optimization']
    loss_function = config['loss_function'] if 'loss_function' in config.keys() else 'fl'
    set_global_param('loss_function', loss_function)

    hyperparams = {
        'learning_rate': hp.uniform('learning_rate', 0.00001, 0.001),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256, 512]),
        'dropout_p': hp.uniform('dropout_p', 0, 0.5)
    }

    if loss_function == 'fl':
        hyperparams.update({
        'l_alpha': hp.uniform('l_alpha', 0.4, 1),
        'l_gamma': hp.uniform('l_gamma', 1, 3.5),
        'l_reduction': hp.choice('l_reduction', ['mean', 'sum']),
        })

    set_global_params(config)
    trials = Trials()
    best_config = fmin(train_for_opt, hyperparams, algo=tpe.suggest, trials=trials, max_evals=max_evals)

    return best_config, trials


def train_for_opt(params):
    print(params)
    # print(global_params)
    create_model_function = get_global_param('create_model_function')
    model_name = get_global_param('model_name')

    train_graph = get_global_param('train_graph')
    val_graph = get_global_param('validation_graph')
    max_epochs = get_global_param('num_epochs')
    patience = get_global_param('patience')
    favorite_device = get_global_param('favorite_device')
    optimizing_metric = get_global_param('optimizing_metric')
    loss_function = get_global_param('loss_function')

    dropout_p = params['dropout_p']

    model = create_model_function(model_name, train_graph, dropout_p)
    val_mask = val_graph['review'].val_mask

    batch_size = params['batch_size']
    learning_rate = params['learning_rate']

    create_dataloader = get_global_param('create_dataloader_function')
    train_dataloader = create_dataloader(train_graph, batch_size=batch_size)
    val_dataloader = create_dataloader(val_graph, val_mask, batch_size=batch_size)

    if loss_function == 'fl':
        l_alpha = params['l_alpha']
        l_gamma = params['l_gamma']
        l_reduction = params['l_reduction']
        model, history, metrics = train(model=model,
                                        train_graph=train_graph,
                                        train_dataloader=train_dataloader,
                                        validation_graph=val_graph,
                                        validation_dataloader=val_dataloader,
                                        num_epochs=max_epochs,
                                        fl_alpha=l_alpha,
                                        fl_gamma=l_gamma,
                                        fl_reduction=l_reduction,
                                        patience=patience,
                                        favorite_device=favorite_device,
                                        learning_rate=learning_rate,
                                        loss_function=loss_function,
                                        return_metrics=[optimizing_metric])
    else:
        model, history, metrics = train(model=model,
                                        train_graph=train_graph,
                                        train_dataloader=train_dataloader,
                                        validation_graph=val_graph,
                                        validation_dataloader=val_dataloader,
                                        num_epochs=max_epochs,
                                        patience=patience,
                                        favorite_device=favorite_device,
                                        learning_rate=learning_rate,
                                        loss_function=loss_function,
                                        return_metrics=[optimizing_metric])

    if optimizing_metric == 'f1_1':
        res = {'loss': -metrics['f1_1'], 'status': STATUS_OK, 'history': history, 'model': model}
    else:
        res = {'loss': metrics['val_loss'], 'status': STATUS_OK, 'history': history, 'model': model}
    res.update({'params': params})
    return res


def process_result(trials):
    best_trial = trials.best_trial
    best_model_of_the_best_trial = best_trial['result']['model']

    if isinstance(trials, Trials):
        trials = trials.trials

    rows = []
    models = []
    for item in trials:
        list_epochs = item['result']['history']
        parameters = item['result']['params']
        last_epoch = list_epochs[-1]
        best_epoch_number = last_epoch['ActualBestEpoch']
        best_epoch = list_epochs[best_epoch_number - 1]
        if not best_epoch['EarlyStopping'] == 'best':
            for i in range(len(list_epochs), -1):
                if list_epochs[i]['EarlyStopping'] == 'best':
                    best_epoch = list_epochs[i]

        best_model = item['result']['model']
        best_epoch = adjust_dict(best_epoch)
        trial_id = item['tid']
        best_epoch.update({'trial_id': trial_id})
        best_epoch.update(parameters)
        rows.append(best_epoch)
        models.append(best_model)
    return rows, models, best_model_of_the_best_trial


if __name__ == '__main__':
    print()
