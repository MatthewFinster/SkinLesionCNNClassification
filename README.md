# Skin Lesion Classification with CNNs

This repository contains the implementation of a Convolutional Neural Network (CNN) to classify 7 types of skin lesions using a heavily imbalanced dataset of approximately 3800 labelled images. The project involves data exploration, model experimentation, and performance optimisation.

## Final Metrics
- **Accuracy**: 0.8204
- **Unweighted Average Recall (UAR)**: 0.5766
- **Loss**: 0.5148

## Project Structure
- `reports/`: Documentation, assignment reports, and experiment tracking reports (e.g., Weights & Biases logs).
- `notebooks/`: Jupyter notebooks for data exploration and experimentation.
- `scripts/`: Python scripts organised by functionality:
  - `datasets/`: Scripts for loading and preprocessing the dataset.
  - `train/`: Scripts for training and evaluating the model.
  - `explore/`: Scripts for exploratory data analysis.
  - `models/`: Scripts defining and managing model architectures.

## Key Features
- Tackled a heavily imbalanced dataset using data augmentation and weighted loss functions.
- Experimented with CNN architectures, learning rates, batch sizes, and epochs to optimise performance.
- Incorporated pre-trained models like **ResNet** for transfer learning, significantly improving results.
- Used [Weights & Biases](https://wandb.ai/) for tracking experiments and visualising results.

## Getting Started
This project is part of my portfolio and is intended for demonstration purposes only. The code, reports, and scripts showcase my skills in building and optimising CNNs for image classification tasks. While the dataset cannot be shared due to licensing restrictions, the project structure and reports outline the workflow and methodology.

## Dataset
The dataset consists of approximately 3800 labelled images of skin lesions across 7 classes. Due to licensing restrictions, the dataset is not included in this repository. If you'd like to replicate the project, consider using publicly available datasets such as [ISIC Archive](https://www.isic-archive.com/).

## Tools and Frameworks Used
- **Python Libraries**: TensorFlow, PyTorch, NumPy, Matplotlib, Scikit-learn
- **Pre-trained Models**: ResNet (via PyTorch)
- **Tracking Tools**: Weights & Biases for experiment tracking
- **Jupyter Notebooks**: For data exploration and initial experimentation

## License
This repository is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivs (CC BY-NC-ND)](https://creativecommons.org/licenses/by-nc-nd/4.0/) license.

You are free to view the material in this repository for personal or portfolio purposes. However:
- Redistribution, modification, or use of the code for academic or commercial purposes is strictly prohibited.
- Proper attribution must be given to the author, Matthew Finster, if referenced.

All reports and documentation are for demonstration purposes only and must not be reused for plagiarism or academic submissions.

## Acknowledgements
- Dataset sourced from [ISIC Archive](https://www.isic-archive.com/).
- [Weights & Biases](https://wandb.ai/) for facilitating experiment tracking.

---
