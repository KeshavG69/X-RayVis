# XRAY_DETECTION

This repository contains the code and resources for an X-ray image classification project. The project uses deep learning techniques to detect abnormalities in X-ray images.

## Project Overview

The primary objective of this project is to develop a deep learning model that can classify X-ray images for medical diagnosis. This involves preprocessing the data, training a convolutional neural network (CNN), and evaluating its performance.

## Project Structure

- `data.py`: Contains functions for data loading and preprocessing, including image augmentation and normalization.
- `dataloader.py`: Manages data loading utilities for training and testing, providing batch processing and shuffling capabilities.
- `main.py`: The main script to run the training and evaluation of the model. It orchestrates the workflow by calling necessary functions from other modules.
- `model.py`: Defines the deep learning model architecture, specifically a convolutional neural network designed for image classification.
- `utils.py`: Contains utility functions such as performance metrics, plotting functions, and model saving/loading mechanisms.
- `CUSTOM_XRAY_DETECTION_BEST.pth`: Pre-trained model weights for the best performing model.
- `output.png`: Accuracy graph of the classifier model, showcasing the model's performance over training epochs.

## Dataset

The dataset used for this project is the [Unifesp X-ray Body Part Classifier](https://www.kaggle.com/competitions/unifesp-x-ray-body-part-classifier/data) available on Kaggle. It contains X-ray images labeled by body part.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KeshavG69/XRAY_DETECTION.git
   cd XRAY_DETECTION
   ```

2.Install the required packages:
bash
```
pip install -r requirements.txt
```

## Usage

1. Prepare the dataset:
  Organize the dataset into appropriate directories for training and testing.
  Update the data paths in data.py as needed.
2. Run the training script:
bash
```
python main.py
```
3. Evaluate the model and view the results. The accuracy and loss during training will be displayed, and the final accuracy graph will be saved as output.png.

## Results

The model achieved an accuracy of 92.5% on the test dataset. The accuracy graph of the classifier model is shown below:


![Accuraccy Graph](https://github.com/KeshavG69/XRAY_DETECTION/blob/main/output.png)
## Contributing

Contributions are welcome. Please submit a pull request or open an issue for any improvements or suggestions.

## License

This project is licensed under the MIT License.
