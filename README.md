# Image Classification with ANN and CNN

This project demonstrates **Image Classification** on the **CIFAR-10 dataset** using both an **Artificial Neural Network (ANN)** and a **Convolutional Neural Network (CNN)**. The goal is to compare performance between a simple fully-connected neural network and a more powerful convolutional model.

---

## 📌 Project Overview

* **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (60,000 images across 10 classes)
* **Techniques**:

  * ANN (fully connected network)
  * CNN (convolutional layers with pooling)
* **Framework**: TensorFlow / Keras
* **Goal**: Classify images into one of 10 categories.

---

## 📂 Repository Structure

```
├── ImageClassification.ipynb   # Jupyter Notebook with full code
├── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/image-classification.git
cd image-classification
```

### 2. Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install tensorflow numpy matplotlib
```

### 3. Run the Jupyter Notebook

```bash
jupyter notebook ImageClassification.ipynb
```

---

## 🧠 Model Architectures

### Artificial Neural Network (ANN)

* **Input Layer**: Flatten CIFAR-10 image (32×32×3)
* **Hidden Layers**:

  * Dense (3000 neurons, ReLU)
  * Dense (1000 neurons, ReLU)
* **Output Layer**: Dense (10 neurons, sigmoid)
* **Optimizer**: Adam
* **Loss Function**: Sparse categorical crossentropy
* **Training**: 5 epochs

### Convolutional Neural Network (CNN)

* **Convolution + Pooling layers** for feature extraction
* **Dense layers** for classification
* **Better accuracy** compared to ANN due to spatial feature learning

---

## 📊 Results

* ANN achieves moderate accuracy but struggles with complex image features.
* CNN significantly improves classification accuracy, showing the effectiveness of convolutional layers for image tasks.

---

## 📈 Example Outputs

* Training loss and accuracy plots
* Sample images with predicted vs actual labels

---

## 🛠 Future Improvements

* Data augmentation to reduce overfitting
* Experiment with deeper CNN architectures (ResNet, VGG)
* Hyperparameter tuning (learning rate, batch size, optimizers)

---

## 🤝 Contributing

Contributions are welcome! Please fork this repo and create a pull request.

---

## 📜 License

This project is licensed under the MIT License.
