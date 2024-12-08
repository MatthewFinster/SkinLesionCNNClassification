from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import tqdm.notebook as tq
import wandb
import torchmetrics
from torch.utils.data import DataLoader

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

# Determine which device on import, and then use that elsewhere.
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)


def plot_confusion_matrix(cm, class_names):
    '''
        cm: the confusion matrix that we wish to plot
        class_names: the names of the classes 
    '''
    cm = cm.cpu().numpy()
    # this normalizes the confusion matrix
    cm = cm.astype(np.float32) / cm.sum(axis=1)[:, None]
    
    df_cm = pd.DataFrame(cm, class_names, class_names)
    ax = sn.heatmap(df_cm, annot=True, cmap='flare')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show()
    
def count_classes(preds):
    '''
    Counts the number of predictions per class given preds, a tensor
    shaped [batch, n_classes], where the maximum per preds[i]
    is considered the "predicted class" for batch element i.
    '''
    pred_classes = preds.argmax(dim=1)
    n_classes = preds.shape[1]
    return [(pred_classes == c).sum().item() for c in range(n_classes)]

def train_epoch(epoch, model, optimizer, criterion, loader, num_classes, device):
    '''
    Train the model on the entire training set precisely once (one epoch).
    Lab 6 has a very similar function.
    '''
    model.train()
    # Initialize metrics
    # TODO: Task 1b - initialize the following torchmetrics metrics
    #       the average epoch loss per example
    train_loss = torchmetrics.MeanMetric().to(device)
    #       accuracy
    train_accuracy = torchmetrics.Accuracy(task = 'multiclass',
    num_classes=7).to(device)
    #       unweighted average recall
    train_recall = torchmetrics.Recall(task='multiclass',
    num_classes=7, average='macro').to(device)

    for i, (inputs, lbls) in enumerate(loader):
        inputs, lbls = inputs.to(device), lbls.to(device)

        # Update model weights
        # TODO: Task 1b - Use the batch to update the weights of the model
        optimizer.zero_grad() 
        outputs = model(inputs)
        loss = criterion(outputs, lbls) 
        loss.backward()
        optimizer.step() 
        # Accumulate metrics
        # TODO: Task 1b - accumulate each of the 3 metrics you initialized.
        train_loss.update(loss)
        train_accuracy.update(outputs, lbls)
        train_recall.update(outputs, lbls)

        if i == 0:
            print(f"Training Epoch: {epoch}, Batch: {i}")
            print(f"Training - Predictions: {outputs.argmax(dim=1)}")
            print(f"Training - True Labels: {lbls}")
    
    # Calculate epoch metrics, and store in a dictionary for wandb
    # TODO Task 1b - compute the three metrics 
    train_loss_value = train_loss.compute()
    train_accuracy_value = train_accuracy.compute()
    train_recall_value = train_recall.compute()

    print(f'Epoch Loss: {loss.item():.4f}')
    
    metrics_dict = {
      'Loss_train': train_loss_value,
      'Accuracy_train': train_accuracy_value,
      'UAR_train': train_recall_value,
      }

    return metrics_dict

def val_epoch(epoch, model, criterion, loader, num_classes, device):
    '''
    Evaluate the model on the entire validation set.
    '''
    
    # Initialize metrics
    # TODO: Task 1b - initialize the following torchmetrics metrics
    #       the average epoch loss per example
    val_loss = torchmetrics.MeanMetric().to(device)
    #       accuracy
    val_accuracy = torchmetrics.Accuracy(task='multiclass',
    num_classes=7).to(device)
    #       unweighted average recall
    val_recall = torchmetrics.Recall(task='multiclass',
    num_classes=7, average='macro').to(device)

    
    # TODO: Task 1c - initialize a confusion matrix torchmetrics object
    confusion_matrix = torchmetrics.ConfusionMatrix(task ='multiclass', 
    num_classes=7).to(device)
    
    for i, (inputs, lbls) in enumerate(loader):
        inputs, lbls = inputs.to(device), lbls.to(device)
        model.eval()

        # TODO Task 1b - Obtain validation loss (use torch.no_grad())
        with torch.no_grad():
          outputs = model(inputs)
          loss = criterion(outputs, lbls)

        # Accumulate metrics
        # TODO: Task 1b - accumulate each of the 3 metrics you initialized
        val_loss.update(loss)
        val_accuracy.update(outputs, lbls)
        val_recall.update(outputs, lbls)
        #       This will take the loss, outputs and lbls from this batch
        #       to update each of the metric object's internal state. 
    
        # TODO: Task 1c - acculmate confusion matrix 
        confusion_matrix.update(outputs, lbls)

        if i == 0:
          print(f"Validation Epoch: {epoch}, Batch: {i}")
          print(f"Validation - Predictions: {outputs.argmax(dim=1)}")
          print(f"Validation - True Labels: {lbls}")
         
    # Calculate epoch metrics, and store in a dictionary for wandb
    # TODO Task 1b - compute the three metrics 
    val_loss_value = val_loss.compute()
    val_accuracy_value = val_accuracy.compute()
    val_recall_value = val_recall.compute()

    metrics_dict = {
      'Loss_val': val_loss_value,
      'Accuracy_val': val_accuracy_value,
      'UAR_val': val_recall_value,
      }

    # Compute the confusion matrix
    # TODO: Task 1c - compute the confusion matrix and store it in cm
    cm = confusion_matrix.compute()

    return metrics_dict, cm
    


def train_model(model, train_loader, val_loader, optimizer, criterion,
                class_names, n_epochs, project_name, ident_str=None):
                
    num_classes = len(class_names)
    model.to(device)
    
    # Initialise Weights and Biases (wandb) project
    if ident_str is None:
      ident_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model.__class__.__name__}_{ident_str}"
    run = wandb.init(project=project_name, name=exp_name)

    try:
        # Train by iterating over epochs
        for epoch in tq.tqdm(range(n_epochs), total=n_epochs, desc='Epochs'):
            train_metrics_dict = train_epoch(epoch, model, optimizer, criterion,
                    train_loader, num_classes, device)
                    
            val_metrics_dict, cm = val_epoch(epoch, model, criterion, 
                    val_loader, num_classes, device)
            wandb.log({**train_metrics_dict, **val_metrics_dict})
    finally:
        run.finish()

    # Plot confusion matrix from results of last val epoch
    # TODO Task 1c - call plot_confusion_matrix with appropriate arguments.
    plot_confusion_matrix(cm, class_names)
