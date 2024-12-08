import matplotlib.pyplot as plt
import numpy as np

def plot_label_distribution(labels, split, class_names):
    # This function plots the class distribution for a given list of labels
    # The parameters are the following:
    #   labels: list of integers indicating the class of each example 
    #   split: a string specifying the split name you are ploting.
    #          This is used in the title of the graph.
    #          - e.g. 'train' or 'val'
    #   class_names: a list of class names corresponding to each class
    #                - e.g. "MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"
    
    # Given a flat list of integers `labels`, counts how many of each
    counts = [ sum(labels == c) for c in range(len(class_names)) ]

    # Plot a histogram of the distribution --------------------------------
    plt.title(f'{split} distribution')
    plt.bar(class_names, counts)
    plt.xlabel('Class')
    plt.ylabel('Num examples')
    plt.show()

    # Checking the proportionality between the training and validation sets
    # --------------------------------------------------------------------
    
def percentages_distributions(labels, split, class_names):
  print(f'Distributions as percentages - {split} Data')
  counts = [ sum(labels == c) for c in range(len(class_names)) ]
  total = len(labels)
  for i, count in enumerate(counts):
    print(f"{class_names[i]}: {100 / total * count:.2f}%")
    print(f"{class_names[i]}: {count}")
