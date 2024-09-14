import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the confusion matrix from .npy file
confusion_matrix = np.load('MobileNetV2TF/CIFAR100/Training_Results/2024-09-13_16-17-38/confusion_matrix.npy')

# Extract true positives, false positives, true negatives, false negatives
# Assuming binary classification and confusion matrix in the format:
# [[TN, FP],
#  [FN, TP]]
confusion_matrix.print()