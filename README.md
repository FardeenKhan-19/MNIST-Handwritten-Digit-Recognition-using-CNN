# MNIST Handwritten Digit Recognition with a CNN

This repository contains a Python script that builds, trains, and evaluates a Convolutional Neural Network (CNN) for classifying handwritten digits from the famous MNIST dataset. The entire implementation is done using TensorFlow and its high-level Keras API.

[Image of MNIST digit examples]

## 🚀 Project Overview

The primary goal of this project is to provide a clear and simple example of a complete image classification workflow. The script handles the following key steps:
1.  **Data Loading**: Fetches the MNIST dataset directly using `tf.keras.datasets`.
2.  **Preprocessing**: Reshapes the images to be compatible with the CNN input layer and normalizes pixel values to the range [0, 1].
3.  **Model Building**: Defines a sequential CNN model with convolutional, pooling, and dense layers.
4.  **Training**: Compiles and trains the model on the training dataset.
5.  **Evaluation**: Measures the model's performance on the unseen test dataset.
6.  **Visualization**: Plots the training and validation accuracy over epochs and shows a prediction for a sample test image.

---

## 🛠️ Model Architecture

The CNN architecture is designed to learn hierarchical features from the 28x28 pixel images. It consists of the following layers:

| Layer Type          | Details                                       | Output Shape         |
| ------------------- | --------------------------------------------- | -------------------- |
| `Conv2D`            | 32 filters, (3, 3) kernel, ReLU activation    | (None, 26, 26, 32)   |
| `MaxPooling2D`      | (2, 2) pool size                              | (None, 13, 13, 32)   |
| `Conv2D`            | 64 filters, (3, 3) kernel, ReLU activation    | (None, 11, 11, 64)   |
| `MaxPooling2D`      | (2, 2) pool size                              | (None, 5, 5, 64)     |
| `Conv2D`            | 64 filters, (3, 3) kernel, ReLU activation    | (None, 3, 3, 64)     |
| `Flatten`           | -                                             | (None, 576)          |
| `Dense`             | 64 units, ReLU activation                     | (None, 64)           |
| `Dense` (Output)    | 10 units (0-9 classes), Softmax activation    | (None, 10)           |

The model is compiled using the **Adam optimizer** and **sparse categorical crossentropy** as the loss function.

---

## 📦 Requirements

To run this project, you need Python 3 and the following libraries installed:
* `tensorflow`
* `matplotlib`

You can install them using pip:
```bash
pip install tensorflow matplotlib
