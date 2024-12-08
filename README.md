# Skin Lesion Classification with CNNs

This repository contains the implementation of a Convolutional Neural Network (CNN) to classify 7 types of skin lesions using a heavily imbalanced dataset of approximately 3800 labelled images. The project involves data exploration, model experimentation, and performance optimisation.

## Final Metrics
- **Accuracy**: 0.8204
- **Unweighted Average Recall (UAR)**: 0.5766
- **Loss**: 0.5148

## Project Structure
- `datasets/`: Script(s) for loading and preprocessing the dataset.
- `explore/`: Python scripts for exploratory data analysis.
- `train/`: Python scripts for training and evaluation of the model.
- `reports/`: Documentation, assignment reports, and experiment tracking reports (e.g., Weights & Biases logs).

## Key Features
- Tackled a heavily imbalanced dataset using data augmentation and weighted loss functions.
- Experimented with CNN architectures, learning rates, batch sizes, and epochs to optimise performance.
- Incorporated pre-trained models like **ResNet** for transfer learning, significantly improving results.
- Used [Weights & Biases](https://wandb.ai/) for tracking experiments and visualising results.

## Getting Started
This project is part of my portfolio and is intended for demonstration purposes only. The code, reports, and scripts showcase my skills in building and optimising CNNs for image classification tasks. While the dataset cannot be shared due to licensing restrictions, the project structure and reports outline the workflow and methodology.
