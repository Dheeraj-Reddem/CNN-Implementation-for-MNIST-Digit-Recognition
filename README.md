# CNN-Implementation-for-MNIST-Digit-Recognition

Following steps are used to build a Convolutional Neural Network (CNN) from scratch to
classify handwritten digits using the MNIST dataset:
https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
1. Installs the necessary package for fetching the dataset.
The ucimlrepo package provides a convenient interface to access datasets hosted
on the UCI ML Repository directly from within Python code. It allows users to fetch
datasets by specifying their IDs or keywords, making it easier to incorporate these
datasets into machine learning projects without having to manually download and
preprocess them. In the provided code, the fetch_ucirepo function from the
ucimlrepo package is used to fetch the "Optical Recognition of Handwritten Digits"
dataset, which is one of the datasets available on the UCI ML Repository.
2. First of all, necessary python libraries are imported as this task is going to be done in
Python Programming Language. Some fundamentals libraries are:
a. ucimlrepo: This library is used to fetch datasets from the UCI Machine
Learning Repository. It provides a convenient way to access datasets for
machine learning tasks without manually downloading them from the
repository.
b. numpy: NumPy is used for scientific computing with Python. It provides
support for large, multi-dimensional arrays and matrices, along with a
collection of mathematical functions to operate on these arrays efficiently.
c. tensorflow: TensorFlow is an open-source ML framework and provides tools
and libraries for building and training machine learning models, including
neural networks.
d. tensorflow.keras: Keras is a high-level neural networks API, written in Python
and capable of running on top of TensorFlow. It provides an easy-to-use
interface for building and training deep learning models.
e. matplotlib.pyplot: to visualize data in graphical format
f. sklearn.metrics: Scikit-learn is a machine learning library in Python that
provides simple and efficient tools for data mining and data analysis. The
metrics submodule contains various functions for evaluating the
performance of machine learning models, such as accuracy, precision, F1
score, ROC AUC score, and confusion matrix.
g. sklearn.model_selection: This submodule of scikit-learn provides utilities for
splitting data into training and test sets, KFold, cross-validation, and
parameter tuning.
3. Then, fetches the optical recognition of handwritten digits dataset from the UCI
repository, extracts the features (X) and targets (y), and converts them to NumPy
arrays.
4. The data is reshaped to have a shape suitable for convolutional neural networks
(CNNs), and the target labels are one-hot encoded.
5. Then, plots a subset of training images to visualize the handwritten digits.
6. Then we defined a CNN architecture with convolutional layers followed by max-
pooling layers, fully connected layers, and softmax activation for multiclass
classification.
7. On next step, the model is compiled with the Adam optimizer, categorical cross-
entropy loss, and accuracy metric. Then it's trained on the training data with a batch
size of 64 for 10 epochs.
8. After that, we plot the training and validation accuracy over epochs to visualize the
model's training progress.
9. For evaluation, the model is evaluated on the test data, and the test accuracy is
printed. F1 score, precision, and AUC ROC are calculated using scikit-learn metrics
functions. Performed K-fold cross-validation to assess the model's performance on
different subsets of the data.
10. Finally, the confusion matrix is computed and displayed to evaluate the model's
performance on each class.
Overall, this code provides a thorough process for building, training, evaluating,
validating and performance analysis of a CNN model for digit recognition using the
MNIST dataset.
