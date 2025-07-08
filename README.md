# 🤖 Multilayer Perceptron (MLP) – Neural Network From Scratch in Go (Golang)

This project is a complete **manual implementation of a Multilayer Perceptron (MLP)** neural network written in **Go (Golang)** — without using any external machine learning or deep learning frameworks.

It demonstrates a deep understanding of neural networks, numerical computing, and algorithm design in a statically typed compiled language not traditionally associated with AI development.

---
##### A Bit of History
	-   Neural networks date back to the 1940s (e.g., Turing's "B-type machines")
	-   The **Perceptron** was proposed by **Frank Rosenblatt in 1957**.
    -   The **Multilayer Perceptron** (MLP) later emerged as a powerful tool once backpropagation was introduced.

---
## 🧠 Project Overview

The goal is to classify data into multiple categories using a feedforward neural network with one or more hidden layers.

The implementation includes:

- Custom matrix operations  
- Configurable network architecture (input, hidden, output layers)  
- Forward and backward propagation logic  
- Manual gradient descent optimizer  
- Common activation functions (ReLU, Sigmoid, Softmax)  
- Cross-entropy loss function  
- Evaluation metrics: accuracy, confusion matrix

---

## 🚀 Why Go?

Go is known for its speed, simplicity, and concurrency support — but it's rarely used in AI. This project pushes its boundaries by applying it to machine learning tasks.

**Why this matters**:

- Shows ability to implement low-level ML logic in a performant, typed language  
- Demonstrates language-agnostic ML proficiency  
- Useful in systems where performance, deployment, and simplicity are key

---

## 🧩 Architecture Overview

### 1. Data Processing  
- CSV reader and parser  
- Min-max normalization and label encoding  
- Split into training and testing sets

### 2. Model Architecture  
- Fully connected layers  
- Forward propagation using dot products  
- Activation: Sigmoid, ReLU, or Softmax  
- Configurable layer sizes

### 3. Training Process  
- Manual backpropagation implementation using chain rule  
- Loss gradient computation (categorical cross-entropy)  
- Weight and bias updates with learning rate tuning  
- Epoch loop with loss and accuracy tracking

### 4. Evaluation  
- Confusion matrix and per-class accuracy  
- Final predictions and model export (if needed)

---

## 💡 Key Features

- ✅ Pure Go — No external ML/AI libraries  
- ✅ Full forward & backward pass logic  
- ✅ Modular and scalable architecture  
- ✅ Reproducible and testable

---

## 🖼️ Optional Visualizations (Python or Go-based)

If needed, performance curves (loss/accuracy) can be generated using:

- Matplotlib (via exported `.csv`)
- Gonum / SVGo (for native Go plots)

---

