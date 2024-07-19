
# [Floodnet Classification]
- Dataset: https://drive.google.com/drive/folders/1sZZMJkbqJNbHgebKvHzcXYZHJd6ss4tH  
- Model: Resnet50

# Floodnet Classification - Post Flood Scene Understanding

## Overview

This project focuses on using computer vision techniques to understand and classify post-flood scenes. Utilizing the Floodnet dataset, which contains images from post-Hurricane Harvey, we aim to train a model to classify regions as flooded or non-flooded. This can significantly aid disaster management response efforts by enabling quick identification and action in affected areas.

## Directory Structure

The repository is organized as follows:

- **Code/**: Contains all the scripts and notebooks used for data preprocessing, model training, and evaluation.
- **Presentation/**: Slides and materials used for presenting the project.
- **Proposal/**: Initial project proposal outlining objectives, methodology, and expected outcomes.
- **Report/**: Comprehensive report detailing the project, methodology, results, and conclusions.

## Abstract

Visual scene understanding is crucial in disaster management response. The Floodnet dataset, collected using UAVs after Hurricane Harvey, provides images of both flooded and non-flooded regions. This project analyzes the classification task using transfer learning to efficiently and quickly classify images. By identifying flooded areas, we can expedite the dispatch of aid to required locations.

## Objectives

1. **Understand and preprocess the Floodnet dataset**.
2. **Implement a classification model using ResNet50 and transfer learning**.
3. **Address challenges such as class imbalance and high-resolution images**.
4. **Evaluate model performance and optimize for accuracy and computational efficiency**.

## Introduction

The increase in natural disasters due to climate change necessitates advanced response mechanisms. Visual scene understanding through computer vision can greatly enhance the effectiveness of post-flood responses. This project leverages the Floodnet dataset to build a classification model that distinguishes between flooded and non-flooded regions using transfer learning.

## Related Works

### Image Classification

Image classification involves assigning labels to images based on their content. For this project, we use single-label classification to categorize images as flooded or non-flooded. ResNet50 is chosen for its efficiency and ability to mitigate issues like vanishing gradients through skip connections.

### Semantic Segmentation

Semantic segmentation labels each pixel in an image with a corresponding class. While not the primary focus of this project, segmentation can provide detailed information about the scene, complementing classification efforts.

### Visual Question Answering (VQA)

VQA combines computer vision and natural language processing to answer questions about images. This can be beneficial in disaster response scenarios, allowing responders to obtain specific information about affected areas.

## Methodology

### Data and Preprocessing

- **Dataset**: Floodnet dataset with 2343 images, labeled for classification and segmentation tasks.
- **Class Imbalance**: Only 51 images are labeled as flooded, while 347 are non-flooded. We use weighted sampling to address this imbalance.
- **Image Resizing**: Images are resized to 300x400 to manage computational load.
- **Data Augmentation**: Techniques like resizing, shifting, cropping, and flipping are used to increase the number of training samples.

### Model and Training

- **Model**: ResNet50 with an added input layer and a final sigmoid activation layer for binary classification.
- **Compilation Parameters**: Adam optimizer, binary cross-entropy loss, and accuracy metric.
- **Training**: Model is trained on the preprocessed dataset, with a separate validation set and 10% of the data reserved for testing.

## Results

The model achieved high accuracy and computational efficiency, with the following performance metrics:

- **Training Accuracy**: High accuracy observed within a few epochs due to the pretrained architecture of ResNet50.
- **Validation Accuracy**: Consistently high validation accuracy.
- **Test Accuracy**: Approximately 80%.
- **F1 Score**: 87%, indicating a good balance between precision and recall.

## Conclusion

This project successfully demonstrates the application of transfer learning for post-flood scene classification using the Floodnet dataset. The model's performance highlights the potential for rapid and accurate disaster response, aiding in efficient resource allocation and timely aid delivery.

## Future Work

Future efforts can focus on:

- **Enhancing model robustness**: Incorporating more data and exploring advanced augmentation techniques.
- **Segmentation and VQA tasks**: Extending the project to include detailed scene segmentation and visual question answering for comprehensive scene understanding.
- **Real-world deployment**: Implementing the model in real-time systems for immediate disaster response.

## Contact

For any questions or contributions, please reach out to Arjun Arora at Technische Universität Braunschweig.
