# Garbage-classification
This repository ([Garbage Classifier Model](https://github.com/R-Ahmadi01/Garbage-classification/blob/main/CLIP-Garbage-classification.ipynb)) provides a comprehensive pipeline for classifying garbage images into specific categories by leveraging the power of CLIP (Contrastive Language–Image Pretraining) and a custom neural network classifier. The project focuses on combining both image and text features for enhanced classification accuracy across four categories: Blue, TTR, Green, and Black.


## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Architecture](#Architecture)
5. [Results](#Results)
6. [Future Work](#Future-Work)

---

 # Introduction
In Canada, particularly in Calgary, waste sorting is a critical part of waste management efforts. The system relies on different color-coded carts for specific types of waste. The black cart is intended for general garbage, the blue cart is for recyclables, and the green cart is for compostable materials such as food waste, soiled cardboard, and other compostable items. Additionally, certain disposables are color-coded red to indicate special handling. **Figure 1** provides a more detailed guide on how to properly sort waste under this system.

<p align="center">
 <img src="https://github.com/user-attachments/assets/6a56224d-710e-4634-a3c8-4670dc3d0a58" alt="Picture1" width="360"/>
</p>
<p align="center">
   <em>Figure 1. This guide illustrates Calgary's waste sorting system, including the proper disposal of garbage, recyclables, compostable materials, and other categorized items based on color-coded carts.</em>
</p>




In this project, we utilize OpenAI's CLIP model, which integrates both vision and language models, to extract rich semantic features from images and their corresponding textual descriptions. By fine-tuning a small portion of the CLIP model and training a custom classifier, we achieve a highly accurate fusion-based classification model for garbage categorization.

-------

# Dataset

The dataset includes images classified into four groups: `Blue`, `TTR`, `Green`, and `Black`. Each image filename serves as its text description for simplicity.

- **Data Augmentation**: Applied random cropping, flipping, color jitter, and normalization to improve model robustness.
- **Train, Validation, and Test Splits**: The dataset is divided into training, validation, and testing folders for a structured evaluation of model performance.

| Dataset Split | Number of Images |
|---------------|------------------|
| Train         |     10200        |
| Validation    |     1800         |
| Test          |     3431         |

*Table 1. Dataset Distribution.*

---
# Data Preprocessing
Effective preprocessing is crucial for improving model performance and ensuring consistency across training, validation, and test data. In this project, we use specific data transformations for training and evaluation phases.

**Training Data Augmentation**
To make the model more robust, several data augmentation techniques are applied to the training images:
- **Random Resized Crop**: Randomly crops the image to a size of 224x224 pixels. This transformation helps the model generalize better by introducing variations in scale.
Random Horizontal and Vertical Flip: Randomly flips the image horizontally and vertically with a 50% probability, making the model invariant to orientation changes.
- **Color Jitter**: Adjusts brightness, contrast, saturation, and hue with small variations, making the model less sensitive to lighting conditions.
- **Normalization**: Applies mean and standard deviation normalization, which centers pixel values and brings them into a standard range that matches the model’s expected input format. The values used are:
    - Mean: [0.48145466, 0.4578275, 0.40821073]
    - Standard Deviation: [0.26862954, 0.26130258, 0.27577711]
  
**Validation and Test Data Transformations**
For validation and testing, we apply minimal transformations to keep the input consistent and avoid altering the data distribution:

- **Resize and Center Crop**: Resizes the image to 224 pixels and then crops it from the center to ensure a fixed input size.
- **Normalization**: The same mean and standard deviation normalization as applied in training.

**Text Preprocessing**
Each image is associated with a textual description derived from the filename:

- **Filename Parsing**: The filename (without extension) is used as the text description.
- **Tokenization**: Text descriptions are tokenized using CLIP’s tokenizer to convert them into embeddings compatible with the model.


This preprocessing pipeline ensures that both image and text data are formatted consistently, allowing for effective fusion of features in the model.


---

# Architecture

The model architecture in this project consists of two main components: the CLIP feature extractor and a custom classifier (see **figure 2**). 

1. **CLIP Feature Extractor**
   We use the CLIP model (`ViT-B/32` variant) for extracting rich image and text 
   embeddings:
   
-   **Pre-Trained Layers**: All layers in the CLIP model are pre-trained on paired 
      image and text data. For efficient transfer learning, all layers are frozen 
      except for the last transformer block.
-   **Embedding Dimensions**: Both image and text embeddings are 512-dimensional, and 
      these embeddings are concatenated to form a 1024-dimensional input to the 
      classifier.
2.**Custom Classifier**
   The classifier is a fully connected neural network that processes the fused image 
   and text embeddings to classify the input into one of four categories: Blue, TTR, 
   Green, or Black.

- **Layer Structure**:
   - **Batch Normalization**: Applied separately to the 512-dimensional image and text feature vectors before they are concatenated. This normalization step helps stabilize training and improves convergence.
   - **Input Layer**: Takes a concatenated 1024-dimensional vector (512 from batch-normalized image features + 512 from batch-normalized text features).
   - **Hidden Layers**: 
        - **Layer 1**: Fully connected, 1024 units, ReLU activation, Dropout (0.5)
        - **Layer 2**: Fully connected, 512 units, ReLU activation, Dropout (0.5)
        - **Layer 3**: Fully connected, 256 units, ReLU activation
   - **Output Layer**: Fully connected layer with 4 units (one per class).

- **Dropout Regularization**: A dropout rate of 0.5 is applied in the hidden layers to reduce overfitting.

**Hyperparameters for Training**
The training process is configured with the following hyperparameters to optimize learning and prevent overfitting:

- **Optimizer**: We use the AdamW optimizer, with a two-part learning rate:
   - **Classifier Parameters**: Learning rate of 0.001
   - **Last Transformer Block in CLIP**: Learning rate of 1e-6 to fine-tune the last layer in the CLIP model carefully.
- **Learning Rate Scheduler**: A StepLR scheduler with `step_size=2` and `gamma=0.5`, reducing the learning rate every 2 epochs to promote stability as training progresses.
- **Loss Function**: `CrossEntropyLoss`, suitable for multi-class classification.
- **Batch Size**: 64 for training, validation, and testing.
- **Early Stopping**: Monitored on validation loss with a patience of 5 epochs to avoid overfitting.
- **Epochs**: Training is set for a maximum of 20 epochs, though early stopping may halt training earlier if validation performance stabilizes.

<p align="center">
 <img src="https://github.com/user-attachments/assets/df831230-c806-449e-bab8-eeb19cde582d" alt="Model Structure" width="720"/>
</p>
<p align="center">
  <em>Figure 2. Model Architecture: Fusion of CLIP embeddings and custom classifier network.</em>
</p>


---

# Results
**Training and Validation Performance**

This table provides an overview of the model's training and validation accuracy across epochs, with metrics recorded at both the start and final epochs. Monitoring these metrics helps assess the model's ability to generalize and avoid overfitting.

*Table 2 Training and validation accuracy across epochs.*
| Epoch |   Training Loss   |  Validation Loss   |
|-------|-------------------|--------------------|
| 1st   |      0.4355       |      0.2836        |   
| 2nd   |      0.2735       |      0.2810        |   
| 3rd   |      0.2121       |      0.2634        |   
| Final |      0.1740       |      0.2521        |   



**Test Performance**

The model achieved an overall test accuracy of ***84.6%***. Figure 3 shows the confusion matrix, which highlights the performance of the model in distinguishing between the four classes. The confusion matrix offers insights into the class-level predictions, showing where the model performs well and where there may be misclassifications.


<p align="center">
 <img src="https://github.com/user-attachments/assets/91925bd8-bd83-4789-abc2-c77bf461f629" alt="Confusion Matrix" width="720"/>
</p>
<p align="center">
    <em> Figure 3 Confusion Matrix on Test Data.</em>
</p>


The **table 3** presents the model's performance metrics for each class, including **precision**, **recall**, and **F1-score**. These metrics offer insights into how accurately the model identifies each class. Higher values indicate stronger classification accuracy, with the F1-score providing a balanced measure of precision and recall.

*Table 3 Model Performance Metrics Across Classes.*
| Class | Precision |   Recall  |  F1-Score    |
|-------|-----------|-----------|--------------|
| Blue  |    0.80   |    0.91   |    0.85      |
| TTR   |    0.90   |    0.77   |    0.83      |
| Green |    0.92   |    0.94   |    0.93      |
| Black |    0.78   |    0.73   |    0.76      |

---

# Future Work

Potential enhancements for this project include:
- **Hyperparameter Tuning**: Experiment with different optimizers and learning rates for further performance improvements.
- **Model Extensions**: Test advanced transformer-based models like CLIP ViT-L for richer feature extraction.
- **Additional Data**: Expand the dataset with more diverse images to improve model generalization.
