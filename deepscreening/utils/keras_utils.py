# coding: utf-8
import json
from keras import optimizers
from ..tripletloss import TripletLoss
from ..metrics import nearest_label_accuracy

def arange_metrics(params={}):
    metric_layers = params.get("metric_layers", [])
    metric_dict = {}
    for layer in metric_layers:
        metric = params.get(f"metric_{layer}")
        if metric is None:
            continue
        # Deal with self-made metrics.
        elif metric == "nearest_label_accuracy":
            metric = nearest_label_accuracy
        metric_dict[layer] = metric
        print(f"[metric] {layer} : {metric}")
    return metric_dict

def arange_losses(params={}):
    loss_layers = params.get("loss_layers", [])
    loss_dict = {}
    for layer in loss_layers:
        loss = params.get(f"loss_{layer}")
        if loss is None:
            continue
        # Deal with self-made loss function.
        elif loss == "tripletloss":
            loss_kwargs = params.get(f"loss_{layer}_kwargs")
            loss = TripletLoss(**loss_kwargs)
        loss_dict[layer] = loss
        print(f"[loss] {layer} : {loss}")
    return loss_dict

def arange_optimizers(params={}):
    optimizer        = params.get("optimizer")
    optimizer_kwargs = params.get("optimizer_kwargs")
    optimizer        = optimizers.__dict__.get(optimizer)(**optimizer_kwargs)
    print(f"optimizer: {optimizer}\n{json.dumps(optimizer_kwargs, indent=2)}")
    return optimizer
