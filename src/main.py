import argparse
import configparser
import datetime
import itertools
import os
import pickle
import warnings

import pandas as pd

from hyperparams_optimization import *
from src.Logger import Logger
from src.OsdGraphPyg import OsdGraphPyg
from src.learning_utils import *
from src.model_utils import create_model


def main(args):
    warnings.filterwarnings('ignore')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(42)

    # Parametri dati per creazione grafo
    path_user_data = args.path_user_data
    path_item_data = args.path_item_data
    path_review_data = args.path_review_data
    path_edges = args.path_edges

    # Parametri per modalità di esecuzione
    mode = args.mode

    loss_function = args.loss_function

    # Parametri per la focal loss
    fl_alpha = args.fl_alpha
    fl_gamma = args.fl_gamma
    fl_reduction = args.fl_reduction

    # Parametri per le feature da utilizzare e da ignorare
    num_fake_user_feature = args.num_fake_user_feature
    num_fake_item_feature = args.num_fake_item_feature
    ignore_review_feature = args.ignore_review_feature
    ignore_user_feature = args.ignore_user_feature

    # Parametri per la cartella di output
    name_folder_experiment = args.name_folder_experiment
    post_text_folder = args.post_text_folder

    # Parametri per il training/test/validation
    batch_size = args.batch_size
    favorite_device = args.favorite_device
    dropout_p = args.dropout_p

    retrain_on_val = args.retrain_on_val

    if mode == 'train':
        print("MODALITA' TRAIN")
        max_epochs = args.max_epochs
        patience = args.patience
        model_name = args.model_name
        learning_rate = args.learning_rate

        logger = Logger()
        logger.add(str(args))

        current_path = os.path.abspath(__file__)
        project_path = os.path.abspath(os.path.join(current_path, "../../"))
        path_experiment = project_path + '/experiment/' + name_folder_experiment + '/exp_' + \
                          str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + post_text_folder + '/')
        os.makedirs(path_experiment, exist_ok=True)

        dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                           num_fake_item_feature=num_fake_item_feature, num_fake_user_feature=num_fake_user_feature,
                           ignore_review_feature=ignore_review_feature, ignore_user_feature=ignore_user_feature)

        print("Grafo di training")
        print(dati.get_train_graph())
        logger.add('Training graph\n' + str(dati.get_train_graph()))

        print("Grafo di validation")
        print(dati.get_val_graph())
        logger.add('Validation graph\n' + str(dati.get_val_graph()))

        print("Grafo di test")
        print(dati.get_test_graph())
        logger.add('Test graph\n' + str(dati.get_test_graph()))

        train_graph = dati.get_train_graph()
        val_graph = dati.get_val_graph()
        val_mask = val_graph['review'].val_mask
        test_graph = dati.get_test_graph()
        test_mask = test_graph['review'].test_mask

        train_dataloader = create_dataloader(train_graph, batch_size=batch_size)
        val_dataloader = create_dataloader(val_graph, val_mask, batch_size=batch_size)
        test_dataloader = create_dataloader(test_graph, test_mask, batch_size=batch_size)

        model = create_model(model_name, train_graph, dropout_p)

        print(model)
        logger.add('Model:\n' + str(model))
        logger.add('Focal loss alpha: ' + str(fl_alpha))
        logger.add('Focal loss gamma: ' + str(fl_gamma))
        logger.add('Focal loss reduction: ' + str(fl_reduction))

        # print_model(model, train_graph, train_dataloader)
        # model = ModelCustom(train_graph, 13, 773)

        cp = os.getcwd()
        os.chdir(path_experiment)

        model, history, metrics = train(model=model, train_graph=train_graph, train_dataloader=train_dataloader,
                                        validation_graph=val_graph, validation_dataloader=val_dataloader,
                                        num_epochs=max_epochs,
                                        fl_alpha=fl_alpha, fl_gamma=fl_gamma, fl_reduction=fl_reduction,
                                        patience=patience,
                                        favorite_device=favorite_device, return_metrics=['train_loss', 'epoch'],
                                        learning_rate=learning_rate, loss_function=loss_function)
        history, figures = test(model, test_graph, test_dataloader, fl_alpha=fl_alpha, fl_gamma=fl_gamma,
                                fl_reduction=fl_reduction, favorite_device=favorite_device, loss_function=loss_function)

        if retrain_on_val:
            trainval_dataloader = create_dataloader(val_graph, batch_size=batch_size)
            assert args.retrain_mode == 'loss' or args.retrain_mode == 'epochs'
            retrain_mode = args.retrain_mode

            retrained_model, re_history, re_metrics = retrain(retrain_mode,
                                                              trainval_dataloader,
                                                              val_graph,
                                                              model=model,
                                                              train_loss=metrics['train_loss'],
                                                              val_graph=val_graph,
                                                              val_dataloader=val_dataloader,
                                                              num_epochs=max_epochs,
                                                              fl_alpha=fl_alpha,
                                                              fl_gamma=fl_gamma,
                                                              fl_reduction=fl_reduction,
                                                              favorite_device=favorite_device,
                                                              learning_rate=learning_rate,
                                                              create_model_function=create_model,
                                                              model_name=model_name,
                                                              dropout_p=dropout_p,
                                                              loss_function=loss_function,
                                                              epochs_for_retrain=metrics['epoch']
                                                              )
            re_history, re_figures = test(retrained_model, test_graph, test_dataloader, fl_alpha=fl_alpha,
                                          fl_gamma=fl_gamma,
                                          fl_reduction=fl_reduction, favorite_device=favorite_device,
                                          history=re_history, loss_function=loss_function)

            torch.save(retrained_model, 'retrained_' + retrained_model.model_name() + ".m")
            torch.save(retrained_model.state_dict(), 'retrained_' + retrained_model.model_name() + "_state_dict.txt")
            for key in re_figures.keys():
                re_figures[key].savefig(key + '_retrained.png', format='png')
            logger.add(re_history)

        torch.save(model, model.model_name() + ".m")
        torch.save(model.state_dict(), model.model_name() + "_state_dict.txt")
        for key in figures.keys():
            figures[key].savefig(key + '.png', format='png')
        logger.add_list(history)

        logger.save('log.txt')

        os.chdir(cp)
    elif mode == 'test':
        print("MODALITA' TEST")
        model_path = args.model
        if not os.path.isfile(model_path):
            raise Exception('Il file specificato non esiste')
        model = torch.load(model_path)
        path_experiment = os.path.dirname(model_path) + '/'

        logger = Logger()
        print(model)

        dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                           num_fake_item_feature=num_fake_item_feature, num_fake_user_feature=num_fake_user_feature,
                           ignore_review_feature=ignore_review_feature, ignore_user_feature=ignore_user_feature)

        print("Grafo di test")
        print(dati.get_test_graph())
        logger.add('Test graph\n' + str(dati.get_test_graph()))

        test_graph = dati.get_test_graph()
        test_mask = test_graph['review'].test_mask

        test_dataloader = create_dataloader(test_graph, test_mask, batch_size=batch_size)

        history, figures = test(model, test_graph, test_dataloader, fl_alpha=fl_alpha, fl_gamma=fl_gamma,
                                fl_reduction=fl_reduction, favorite_device=favorite_device, loss_function=loss_function)

        cp = os.getcwd()
        os.chdir(path_experiment)

        for key in figures.keys():
            figures[key].savefig(key + '.png', format='png')
        logger.add_list(history)

        logger.save('test_log.txt')

        os.chdir(cp)
    elif mode == 'hyperopt':
        print("MODALITA' HYPEROPT")
        max_epochs = args.max_epochs
        patience = args.patience
        model_name = args.model_name
        max_iter_optimization = args.max_iter_optimization

        logger = Logger()
        logger.add(str(args))

        current_path = os.path.abspath(__file__)
        project_path = os.path.abspath(os.path.join(current_path, "../../"))
        path_experiment = project_path + '/experiment/' + name_folder_experiment + '/exp_' + \
                          str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + \
                              ('_' + post_text_folder if len(post_text_folder) > 0 else '') + '/')
        os.makedirs(path_experiment, exist_ok=True)

        dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                           num_fake_item_feature=num_fake_item_feature, num_fake_user_feature=num_fake_user_feature,
                           ignore_review_feature=ignore_review_feature, ignore_user_feature=ignore_user_feature)

        print("Grafo di training")
        print(dati.get_train_graph())
        logger.add('Training graph\n' + str(dati.get_train_graph()))

        print("Grafo di validation")
        print(dati.get_val_graph())
        logger.add('Validation graph\n' + str(dati.get_val_graph()))

        print("Grafo di test")
        print(dati.get_test_graph())
        logger.add('Test graph\n' + str(dati.get_test_graph()))

        train_graph = dati.get_train_graph()
        val_graph = dati.get_val_graph()

        best, trials = start_hyperparameters_optimization(create_model, create_dataloader, train_graph, val_graph,
                                                          model_name=model_name, patience=patience,
                                                          num_epochs=max_epochs,
                                                          favorite_device=favorite_device,
                                                          max_iter_optimization=max_iter_optimization,
                                                          optimizing_metric='f1_1',
                                                          loss_function=loss_function)
        rows, models, best_model = process_result(trials)

        test_graph = dati.get_test_graph()
        test_mask = test_graph['review'].test_mask

        test_dataloader = create_dataloader(test_graph, test_mask, batch_size=batch_size)

        history, figures = test(best_model, test_graph, test_dataloader, fl_alpha=fl_alpha, fl_gamma=fl_gamma,
                                fl_reduction=fl_reduction, favorite_device=favorite_device, loss_function=loss_function)

        cp = os.getcwd()
        os.chdir(path_experiment)
        torch.save(best_model, best_model.model_name() + ".m")
        torch.save(best_model.state_dict(), best_model.model_name() + "_state_dict.txt")
        os.mkdir('all_models')
        for key in figures.keys():
            figures[key].savefig(key + '.png', format='png')
        for i in range(len(models)):
            torch.save(models[i], 'all_models/' + models[i].model_name() + "_trial-" + str(rows[i]['trial_id']) + '.m')
        with open('all_models/trials.t', 'wb') as f:
            pickle.dump(trials, f)
            f.close()
        logger.add_list(history)

        logger.save('test_log.txt')

        df_trials = pd.DataFrame(rows)
        df_trials.to_csv('optimization.csv')

        os.chdir(cp)
    elif mode == 'retrain':

        print("WARNING: this mode was not used for GLORIA")

        model_path = args.model
        max_epochs = args.max_epochs
        learning_rate=args.learning_rate
        previous_train_loss = args.retrain_loss_value
        retrain_mode = args.retrain_mode
        model_name = args.model_name

        if not os.path.isfile(model_path):
            raise Exception('Il file specificato non esiste')
        model = torch.load(model_path)
        path_experiment = os.path.dirname(model_path) + '/'

        logger = Logger()
        print(model)

        dati = OsdGraphPyg(path_user_data, path_item_data, path_review_data, path_edges,
                           num_fake_item_feature=num_fake_item_feature, num_fake_user_feature=num_fake_user_feature,
                           ignore_review_feature=ignore_review_feature, ignore_user_feature=ignore_user_feature)

        print("Grafo di validation")
        print(dati.get_val_graph())
        logger.add('Validation graph\n' + str(dati.get_val_graph()))

        val_graph = dati.get_val_graph()
        val_mask = val_graph['review'].val_mask

        val_dataloader = create_dataloader(val_graph, val_mask, batch_size=batch_size)
        trainval_dataloader = create_dataloader(val_graph, batch_size=batch_size)

        retrained_model, history, metrics = retrain(retrain_mode,
                                                    trainval_dataloader,
                                                    val_graph,
                                                    model=model,
                                                    train_loss=previous_train_loss,
                                                    val_graph=val_graph,
                                                    val_dataloader=val_dataloader,
                                                    num_epochs=max_epochs,
                                                    fl_alpha=fl_alpha,
                                                    fl_gamma=fl_gamma,
                                                    fl_reduction=fl_reduction,
                                                    favorite_device=favorite_device,
                                                    learning_rate=learning_rate,
                                                    create_model_function=create_model,
                                                    model_name=model_name,
                                                    dropout_p=dropout_p,
                                                    epochs_for_retrain=max_epochs,
                                                    loss_function=loss_function
                                                    )

        cp = os.getcwd()
        os.chdir(path_experiment)
        torch.save(retrained_model, 'retrained_' + retrained_model.model_name() + ".m")
        torch.save(retrained_model.state_dict(), 'retrained_' + retrained_model.model_name() + "_state_dict.txt")

        logger.add_list(history)

        logger.save('retrain_log.txt')

        os.chdir(cp)

    else:
        raise Exception('Valore non valido per il parametro mode [test,train]')


if __name__ == '__main__':

    parser = configparser.ConfigParser()
    parser.read('config.conf')
    config = parser['classification']

    mode = config.get('mode')
    l_alpha = config.getfloat('l_alpha')
    l_gamma = config.getfloat('l_gamma')
    l_reduction = config.get('l_reduction')
    model_name = config.get('model_name')
    max_epochs = config.getint('max_epochs')
    patience = config.getint('patience')
    num_fake_user_feature = config.getint('num_fake_user_feature')
    num_fake_item_feature = config.getint('num_fake_item_feature')
    batch_size = config.getint('batch_size')
    ignore_review_feature = eval(config.get('ignore_review_feature'))
    ignore_user_feature = eval(config.get('ignore_user_feature'))
    favorite_device = config.get('favorite_device')  # cuda, mps, cpu
    name_folder_experiment = config.get('name_folder_experiment')
    path_user_data = config.get('path_user_data')
    path_item_data = config.get('path_item_data')
    path_review_data = config.get('path_review_data')
    path_edges = config.get('path_edges')
    model = config.get('model') # path to serialized model to test
    dropout_p = config.getfloat('dropout_p')
    max_iter_optimization = config.getint('max_iter_optimization')
    learning_rate = config.getfloat('learning_rate')
    loss_function = config.get('loss_function')
    retrain_on_val = config.getboolean('retrain_on_val')
    retrain_mode = config.get('retrain_mode')  # epochs or loss
    retrain_loss_value = config.getfloat('retrain_loss_value')  # solo per mode=retrain

    ignore_configs = []

    args = argparse.Namespace(
        mode=mode,
        model=model,
        fl_alpha=l_alpha,
        fl_gamma=l_gamma,
        fl_reduction=l_reduction,
        max_epochs=max_epochs,
        patience=patience,
        model_name=model_name,
        num_fake_user_feature=num_fake_user_feature,
        num_fake_item_feature=num_fake_item_feature,
        path_user_data=path_user_data,
        path_item_data=path_item_data,
        path_review_data=path_review_data,
        path_edges=path_edges,
        ignore_review_feature=ignore_review_feature,
        ignore_user_feature=ignore_user_feature,
        name_folder_experiment=name_folder_experiment,
        post_text_folder=str(""),
        batch_size=batch_size,
        favorite_device=favorite_device,
        dropout_p=dropout_p,
        max_iter_optimization=max_iter_optimization,
        retrain_on_val=retrain_on_val,
        retrain_mode=retrain_mode,
        retrain_loss_value=retrain_loss_value,
        learning_rate=learning_rate,
        loss_function=loss_function
    )
    print(args)
    main(args)
