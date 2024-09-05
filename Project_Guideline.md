# Assignment 1: Training and Evaluating MobileNETV2 using Tensorflow

| Due Sep 13 by 11:59pm | Points 40 | Submitting a file upload | Available Sep 5 at 10am - Sep 15 at 11:59pm |
| --------------------- | --------- | ------------------------ | ------------------------------------------- |

**Objective:** This assignment aims to train the MobileNetV2 model using TensorFlow on two image datasets. You will evaluate which dataset the model performs better on and provide a justification based on the model's architecture and the nature of the datasets.

## Assignment Tasks:

- Train and evaluate the MobileNetV2 model using TensorFlow on two distinctly different datasets.
  - Image Dataset of your choice: Select a publicly available image dataset (except ImageNet, MNIST handwritten digits).
  - Cifar 100 Dataset: Select the Cifar100 dataset for training and benchmarking your proposed re-modeled architecture of MobileNETV2.
- Preprocess the datasets to ensure they are compatible with the MobileNetV2 model. This includes resizing images, normalizing pixel values, etc.
- Evaluate the MobileNetV2 model’s performance on each dataset using accuracy, precision, recall, and F1-score.
- Compare the final model's performance between the two datasets and determine which dataset yields better results and why.

## BONUS: 5 Points Extra:

- Convert the trained MobileNetV2 model to TensorFlow Lite format.
- Evaluate and compare the performance of the TensorFlow Lite model against the original TensorFlow model in terms of accuracy, model size, and inference speed, and discuss if specific accuracy percentages are degraded.

## Criteria

**Model Training with TensorFlow: (15 points)**

- Load the MobileNetV2 model using TensorFlow and train it separately on each dataset.
- Implement appropriate data augmentation techniques to improve model robustness.
- Your dataset should be split into training, validation, and testing.
- Fine-tune the model's hyperparameters to optimize performance for each dataset.
- Document the training process, including epochs, batch size, learning rate, optimizer used, and other relevant details.
  
**Performance Evaluation: (15 points)**

- Evaluate the performance of the MobileNetV2 model on each dataset using metrics such as accuracy, precision, recall, and F1-score.
- Compare the model's performance between the two datasets and determine which dataset yields better results.

**Justification of Results/Report: (10 points)**

- Analyze and justify why MobileNetV2 performs better on one dataset than the others.
- Consider factors such as the model's architecture, the complexity of the datasets, overfitting/underfitting, and the suitability of the dataset format for MobileNetV2.
- Discuss potential reasons for performance discrepancies and suggest improvements or alternative approaches.

## Submission Format

- You need to submit your Google Collab workbook in .ipynb format.
- You are also required to submit a report (3 pages max).
- You can submit both in a .zip folder or separately through Canvas.

## Report Formatting:

- Cover page (assignment name, group name, and group member names with email IDs)
- Overview of your proposed MobileNETV2 architecture (the training process, including epochs, batch size, learning rate, optimizer used, and other relevant details)
- Evaluation Metrics: Accuracy, Precision, Recall and F1-score S
- Discussion on which dataset yields better evaluation metrics over another.





