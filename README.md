# **Topic Classification with a Feedforward Neural Network**

## **Overview**  
This project, developed for the COM6513 module under the guidance of Instructor Nikos Aletras, focuses on implementing a **Feedforward Neural Network (FNN)** for **topic classification**. The network is trained on text data, utilizing embedding layers and optimization techniques to improve classification accuracy.

## **Key Components**  

### **1. Text Processing**  
- Transformed raw text into input vectors for training.  
- Applied preprocessing techniques such as tokenization, vocabulary filtering, and encoding.  

### **2. Neural Network Architecture**  
- **Input Layer:** One-hot encoding mapped into an embedding weight matrix.  
- **Hidden Layer:** Computes the mean embedding vector of all words in the input, followed by a ReLU activation function.  
- **Output Layer:** Softmax activation for multi-class classification.  

### **3. Training Process**  
- **Optimization:** Used **Stochastic Gradient Descent (SGD)** with **backpropagation** to update network weights.  
- **Loss Function:** Categorical Cross-entropy minimization.  
- **Regularization:** Applied **Dropout** after hidden layers.  
- **Forward & Backward Pass:** Implemented forward propagation to compute intermediate outputs and backward propagation for gradient computation and weight updates.  

### **4. Hyperparameter Tuning**  
- Tuned parameters such as learning rate, embedding size (50, 300, 500), and dropout rate (0.2, 0.5).  
- Evaluated performance through training and validation loss curves.  
- Used tables and graphs to compare different hyperparameter settings.  

### **5. Pretrained Embeddings (GloVe)**  
- Integrated **GloVe embeddings** to initialize the weight matrix instead of random initialization.  
- Implemented weight freezing to prevent updates during training.  
- Analyzed improvements in performance compared to the standard model.  

### **6. Network Extension & Analysis**  
- **Added additional hidden layers** and re-evaluated model performance.  
- Compared different network depths to determine the impact on accuracy and overfitting.  
- Conducted **error analysis** to examine misclassifications and possible improvements.  

## **Technologies Used**  
- **Python** (NumPy, Pandas)  
- **Deep Learning** (Feedforward Neural Network)  
- **Text Processing** (Tokenization, Embeddings)  
- **Optimization** (SGD, Dropout, Cross-Entropy Loss)  