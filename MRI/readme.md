# MRI Brain Tumor Detection

This project focuses on detecting brain tumors from MRI images using deep learning. It leverages a pre-trained ResNet18 model and fine-tunes it for binary classification (tumor/no tumor).

## Features

-   **Data Augmentation and Normalization**: Preprocessing steps to prepare MRI images for training.
-   **Dataset Splitting**: Manual splitting of the dataset into training and validation sets.
-   **Transfer Learning**: Utilization of a pre-trained ResNet18 model from `torchvision.models`.
-   **Model Training**: Training loop with optimization and learning rate scheduling.
-   **Model Evaluation**: Performance assessment using ROC curves and Confusion Matrices.

## Technologies Used

-   Python >= 3.12
-   PyTorch
-   torchvision
-   NumPy
-   Matplotlib
-   scikit-learn

## Installation

This project uses `uv` as the package manager. To set up the environment, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/eloymor/EDA-and-ML.git
    cd EDA-and-ML/MRI
    ```
2.  Install the required packages using `uv`:
    ```bash
    uv uv add torch torchvision matplotlib scikit-learn
    ```

## Usage

The main logic for MRI brain tumor detection is implemented in the `MRI.ipynb` Jupyter Notebook.
