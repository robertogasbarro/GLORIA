import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torchvision.ops import sigmoid_focal_loss
from torch.optim.lr_scheduler import LinearLR
import torch
import torch.nn as nn
from sklearn.metrics import *
import copy
import matplotlib.pyplot as plt
from imblearn.metrics import geometric_mean_score
from utils import adjust_dict


def create_dataloader(graph, *tensors, batch_size=64):
    tensor_dataset = TensorDataset(graph['review'].edge_index.T, graph['review'].label, *tensors)
    random_sampler = RandomSampler(tensor_dataset)
    dataloader = DataLoader(tensor_dataset, sampler=random_sampler, batch_size=batch_size)
    return dataloader


def train(model, train_graph, train_dataloader, validation_graph, validation_dataloader,
          **config):
    """

    :param model:
    :param train_graph:
    :param train_dataloader:
    :param validation_graph:
    :param validation_dataloader:
    :param config: num_epochs, loss_function=['fl','bce'], fl_alpha, fl_gamma, fl_reduction, patience, favorite_device, learning_rate
    train_loss_best (solo per retrain in modalità loss)
    :return:
    """
    print(config)
    num_epochs = config['num_epochs'] if 'num_epochs' in config.keys() else 100
    fl_alpha = config['fl_alpha'] if 'fl_alpha' in config.keys() else 0.25
    fl_gamma = config['fl_gamma'] if 'fl_gamma' in config.keys() else 2
    fl_reduction = config['fl_reduction'] if 'fl_reduction' in config.keys() else 'none'
    patience = config['patience'] if 'patience' in config.keys() else 20
    favorite_device = config['favorite_device'] if 'favorite_device' in config.keys() else 'cuda'
    learning_rate = config['learning_rate'] if 'learning_rate' in config.keys() else 0.0001
    return_metrics = config['return_metrics'] if 'return_metrics' in config.keys() else ['val_loss']
    train_loss_min = config['train_loss_best'] if 'train_loss_best' in config.keys() else -1
    history = config['history'] if 'history' in config.keys() else list()
    enable_early_stopping = config['enable_early_stopping'] if 'enable_early_stopping' in config.keys() else True
    loss_function= config['loss_function'] if 'loss_function' in config.keys() else 'fl'

    print("Start training...")
    device = set_device(favorite_device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = LinearLR(optimizer=opt, start_factor=1, end_factor=0.5, verbose=0, total_iters=50)
    num_train_examples = np.count_nonzero(train_graph['review'].train_mask)
    num_val_examples = np.count_nonzero(validation_graph['review'].val_mask)

    bce_loss = nn.BCELoss()

    best_loss = 10000000
    epochs_no_improve = 0
    best_model = copy.deepcopy(model)
    actual_best_epoch = 0
    for epoch in range(num_epochs):
        record = dict()
        record['Epoch'] = epoch + 1
        model.train()
        training_loss = 0
        true_y_total = []
        y_total = []
        model.to(device)
        for batch in iter(train_dataloader):
            opt.zero_grad()
            graph_gpu = train_graph.to(device)
            batch_edge_index_gpu = batch[0].to(device)
            y = model.forward(graph_gpu, batch_edge_index_gpu)

            true_y = batch[1].to(device)
            if loss_function=='bce':
                loss = bce_loss(y, true_y)
            else:
                loss = sigmoid_focal_loss((1 - y), (1 - true_y), reduction=fl_reduction, alpha=fl_alpha, gamma=fl_gamma)
            loss.backward()
            opt.step()
            training_loss += loss.item()
            true_y_total += list(true_y.cpu().numpy())
            y_total += list(torch.round(y).detach().cpu().numpy())
        before_lr = opt.param_groups[0]["lr"]
        scheduler.step()
        after_lr = opt.param_groups[0]["lr"]

        record['Learning rate'] = before_lr
        # print("Training loss: ", training_loss)
        metrics, _ = calculate_metrics(y_true=true_y_total, y_pred=y_total, text='Training')
        record['Training loss'] = training_loss / num_train_examples
        # record['Training metrics'] = metrics

        record.update(metrics)
        # print(metrics)

        model.eval()
        with torch.no_grad():
            validation_loss = 0
            val_true_y_total = []
            val_y_total = []
            # model.to(device)
            for batch in iter(validation_dataloader):
                graph_gpu = validation_graph.to(device)
                batch_edge_index_gpu = batch[0].to(device)
                batch_mask = batch[2]
                y = model.forward(graph_gpu, batch_edge_index_gpu)[batch_mask]

                true_y = batch[1].to(device)[batch_mask]
                if loss_function == 'bce':
                    loss = bce_loss(y, true_y)
                else:
                    loss = sigmoid_focal_loss((1 - y), (1 - true_y), reduction=fl_reduction, alpha=fl_alpha, gamma=fl_gamma)
                if not np.isnan(loss.cpu()):
                    validation_loss += loss.item()
                val_true_y_total += list(true_y.cpu().numpy())
                val_y_total += list(torch.round(y).detach().cpu().numpy())

            metrics, _ = calculate_metrics(y_true=val_true_y_total, y_pred=val_y_total, text='Validation')
            record['Validation loss'] = validation_loss / num_val_examples
            record.update(metrics)
            if validation_loss < best_loss:
                record['EarlyStopping'] = 'best'
                actual_best_epoch = epoch + 1
            else:
                record['EarlyStopping'] = epochs_no_improve + 1
            record['ActualBestEpoch'] = actual_best_epoch

            history.append(record)
            print_record_train(record)

            if validation_loss/num_val_examples < train_loss_min:
                print("Stop retraining")
                train_graph.cpu()
                validation_graph.cpu()
                model.cpu()
                return model, history, get_metrics(history, epoch+1, return_metrics)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model = copy.deepcopy(model)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience and enable_early_stopping:
                    print("EarlyStopping")
                    train_graph.cpu()
                    validation_graph.cpu()
                    best_model.cpu()
                    return best_model, history, get_metrics(history, actual_best_epoch, return_metrics)
    train_graph.cpu()
    validation_graph.cpu()
    model.cpu()
    return model, history, get_metrics(history, actual_best_epoch, return_metrics)


def get_metrics(history, best_epoch, list_metrics):
    best_epoch_metrics = history[best_epoch - 1]
    metrics = adjust_dict(best_epoch_metrics)
    return_dict = {}
    if 'val_loss' in list_metrics:
        return_dict['val_loss'] = metrics['Validation loss']
    if 'f1_1' in list_metrics:
        return_dict['f1_1'] = metrics['Validation F1'][1]
    if 'train_loss' in list_metrics:
        return_dict['train_loss'] = metrics['Training loss']
    if 'epoch' in list_metrics:
        return_dict['epoch'] = best_epoch
    return return_dict


def calculate_metrics(y_true, y_pred, non_rounded_y_pred=None, text=None, plot_cm=False):
    metrics = dict()
    figs = dict()
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0, average=None)
    metrics['Precision (binary average)'] = precision_score(y_true, y_pred, zero_division=0, average='binary')
    metrics['Precision (weighted average)'] = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0, average=None)
    metrics['Recall (binary average)'] = precision_score(y_true, y_pred, zero_division=0, average='binary')
    metrics['Recall (weighted average)'] = precision_score(y_true, y_pred, zero_division=0, average='weighted')
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0, average=None)
    metrics['F1 (binary average)'] = f1_score(y_true, y_pred, zero_division=0, average='binary')
    metrics['F1 (weighted average)'] = f1_score(y_true, y_pred, zero_division=0, average='weighted')
    metrics['G-mean'] = geometric_mean_score(y_true, y_pred, average=None)
    if non_rounded_y_pred is not None:
        metrics['AUC'], figs['AUC-ROC'] = roc_auc(y_true, non_rounded_y_pred, label=text)
    metrics['Confusion Matrix'] = confusion_matrix(y_true, y_pred).ravel()
    if plot_cm:
        pcm = plotConfusionMatrix(confusion_matrix(y_true, y_pred))
        figs['Confusion Matrix'] = pcm
    metrics['Classification Report'] = classification_report(y_true, y_pred, output_dict=True)

    if text is not None:
        metrics = {text + ' ' + k: v for k, v in metrics.items()}
    return metrics, figs


def test(model, test_graph, test_dataloader, history=None, fl_alpha=0.25, fl_gamma=2, fl_reduction='none',
         favorite_device='cuda', loss_function='fl'):
    print("Testing")
    record = dict()
    device = set_device(favorite_device)

    bce_loss = nn.BCELoss()
    num_test_examples = np.count_nonzero(test_graph['review'].test_mask)
    model.eval()
    model.to(device)
    record['Epoch'] = "TEST"
    with torch.no_grad():
        test_loss = 0
        true_y_total = []
        y_total = []
        y_non_rounded_total = []
        # model.to(device)
        for batch in iter(test_dataloader):
            graph_gpu = test_graph.to(device)
            batch_edge_index_gpu = batch[0].to(device)
            batch_mask = batch[2]
            y = model.forward(graph_gpu, batch_edge_index_gpu)[batch_mask]

            true_y = batch[1].to(device)[batch_mask]
            if loss_function=='bce':
                loss = bce_loss(y, true_y)
            else:
                loss = sigmoid_focal_loss((1 - y), (1 - true_y), reduction=fl_reduction, alpha=fl_alpha, gamma=fl_gamma)
            if not np.isnan(loss.cpu()):
                test_loss += loss.item()
            true_y_total += list(true_y.cpu().numpy())
            y_total += list(torch.round(y).detach().cpu().numpy())
            y_non_rounded_total += list(y.detach().cpu().numpy())

        metrics, figs = calculate_metrics(y_true=true_y_total, y_pred=y_total, text='Test',
                                          non_rounded_y_pred=y_non_rounded_total, plot_cm=True)
        record['Test loss'] = test_loss / num_test_examples
        record.update(metrics)
        if history is None:
            history = list()
        history.append(record)
        print_record_test(record)
    model.cpu()
    test_graph.cpu()
    return history, figs


def print_record_train(record):
    print(f"Epoch: {record['Epoch']}"
          f"\tTrain loss: {record['Training loss']:.4f}"
          f"\tTrain f1: {record['Training F1']}"
          f"\tVal loss: {record['Validation loss']:.4f}"
          f"\tVal f1: {record['Validation F1']}\t {record['EarlyStopping']}")


def print_record_test(record):
    print(f"Test loss: {record['Test loss']}")
    print(f"G-mean: {record['Test G-mean']}")
    print(record['Test Classification Report'])
    print(record['Test Confusion Matrix'])


def format_history(history):
    str = ""
    for row in history:
        str += "Epoch: " + str(row['Epoch'])


def roc_auc(expected, decision_function, label):
    fpr, tpr, thresold = roc_curve(expected, decision_function)
    auc_value = roc_auc_score(expected, decision_function)

    fig = plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(fpr, tpr, linestyle='-', marker='.', label="(auc = %0.4f)" % auc_value)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(label)
    plt.legend()
    plt.show()

    return auc_value, fig


def retrain(mode, trainval_dataloader, trainval_graph, **config):
    assert mode == 'epochs' or mode == 'loss'

    history = config['history'] if 'history' in config.keys() else list()

    if mode == 'loss':
        try:
            previous_train_loss = config['train_loss']
            model = config['model']
            val_graph = config['val_graph']
            val_dataloader = config['val_dataloader']
        except:
            raise Exception("Nella modalita' train_loss e' necessario specificare il modello addestrato (model),"
                            "la loss sul training set (train_loss), il grafo di validation (val_graph) e il dataloader"
                            " sul validation set (val_dataloader)")

        args = config.copy()
        del args['model']
        if 'history' in args.keys(): del args['history']
        del args['train_loss']
        del args['val_graph']
        del args['val_dataloader']

        model, history, metrics = train(model,
                                        trainval_graph,
                                        trainval_dataloader,
                                        val_graph,
                                        val_dataloader,
                                        train_loss_best=previous_train_loss,
                                        history=history,
                                        enable_early_stopping=False,
                                        **args
                                        )
        return model, history, metrics
    elif mode == 'epochs':
        try:
            val_graph = config['val_graph']
            val_dataloader = config['val_dataloader']
            epochs_for_retrain = config['epochs_for_retrain']
            create_model_function = config['create_model_function']
            model_name = config['model_name']
            dropout_p = config['dropout_p']
        except:
            raise Exception("Nella modalita' train_loss e' necessario specificare il modello non addestrato (model),"
                            "la loss sul training set (train_loss), il grafo di validation (val_graph) e il dataloader"
                            " sul validation set (val_dataloader) ed il numero di epoche")

        args = config.copy()
        if 'history' in args.keys(): del args['history']
        del args['val_graph']
        del args['val_dataloader']
        del args['dropout_p']
        del args['num_epochs']
        del args['model']

        model = create_model_function(model_name, trainval_graph, dropout_p)

        model, history, metrics = train(model,
                                        trainval_graph,
                                        trainval_dataloader,
                                        val_graph,
                                        val_dataloader,
                                        history=history,
                                        enable_early_stopping=False,
                                        num_epochs=min(config['num_epochs'], epochs_for_retrain),
                                        **args
                                        )
        return model, history, metrics



def plotConfusionMatrix(cf):
    CMdisplay = ConfusionMatrixDisplay(cf)
    CMdisplay.plot()
    plt.show()
    return CMdisplay.figure_


def set_device(favorite_device):
    return torch.device('cuda' if torch.cuda.is_available() & (favorite_device == 'cuda') else (
        'mps' if torch.backends.mps.is_available() & (favorite_device == 'mps') else 'cpu'))
