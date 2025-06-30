# 🧠 Multilayer Perceptron – Machine Learning Project in Go


## 🧠 Introduction

This project demonstrates how to build a **Multilayer Perceptron (MLP)** from scratch in **Go** (Golang), with no machine learning libraries. The goal is to classify tumor samples as **malignant (M)** or **benign (B)** using the **Wisconsin Breast Cancer Dataset**.

##### A Bit of History
	-   Neural networks date back to the 1940s (e.g., Turing's "B-type machines")
	-   The **Perceptron** was proposed by **Frank Rosenblatt in 1957**.
    -   The **Multilayer Perceptron** (MLP) later emerged as a powerful tool once backpropagation was introduced.

## 🎯 Objectives

-   Implement an MLP **from scratch** in a programming language of your choice.
    
-   Learn the mechanics of **feedforward**, **backpropagation**, and **gradient descent**.
    
-   Apply these concepts to real-world data (breast cancer classification).
    
-   Visualize model performance with **learning curves**.


## ⚙️ General Instructions

-   **No machine learning libraries** (e.g., TensorFlow, PyTorch) allowed.
    
-   Libraries for **linear algebra**, **CSV handling**, and **plotting** are permitted.
    
-   Code must be **clear and modular**
    
-   If using a compiled language, include a `Makefile`.

## 🚧 Mandatory Part

### Dataset

-   **Input**: CSV file with 30 features + 1 diagnosis label (M or B).
    
-   **Action**: Split into **training** and **validation** datasets.
    
-   **Preprocessing** is essential (normalization, label encoding, etc.).

### Model Requirements

-   At least **2 hidden layers**.
    
-   **Softmax** activation in the output layer.
    
-   Use **categorical cross-entropy** as the loss function.
    
-   Must support:
    
    -   Adjustable layer sizes
        
    -   Training hyperparameters (via file or CLI)
        
    -   Visualization of **loss** and **accuracy** over epochs
        

### Programs to Provide

1.  **Dataset Splitter** (with random seed support)
    
2.  **Training Program** (backpropagation + gradient descent + model saving)
    
3.  **Prediction Program** (load model and predict on unseen data)

### Training Output Example
```
Epoch 01/70 - Loss: 0.6882 - Val Loss: 0.6788
Epoch 02/70 - Loss: 0.6501 - Val Loss: 0.6234
...
Epoch 70/70 - Loss: 0.0640 - Val Loss: 0.0474
Model saved to ./models/mlp_model.json
```
