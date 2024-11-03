# Garbage-classification
This repository provides a comprehensive pipeline for classifying garbage images into specific categories by leveraging the power of CLIP (Contrastive Language–Image Pretraining) and a custom neural network classifier. The project focuses on combining both image and text features for enhanced classification accuracy across four categories: Blue, TTR, Green, and Black.


## Table of Contents
1. [Introduction](#introduction)
2. [Architecture](#Architecture)
3. [Dataset](#Dataset)
4. [Installation](#Installation)
5. [Usage](#Usage)
6. [Results](#Results)
7. [Future Work](#Future-Work)
8. [References](#References)

---

 # Introduction
In this project, we utilize OpenAI's CLIP model, which integrates both vision and language models, to extract rich semantic features from images and their corresponding textual descriptions. By fine-tuning a small portion of the CLIP model and training a custom classifier, we achieve a highly accurate fusion-based classification model for garbage categorization.




# Architecture

The model architecture consists of two main components:

1. CLIP Feature Extractor: The CLIP model is pre-trained on large datasets of paired images and texts, providing 512-dimensional embeddings for both modalities. All layers except the last transformer block are frozen during training.
2. Custom Classifier: A fully connected neural network that takes concatenated image and text embeddings as input and classifies them into the four target categories.

<img src="https://github.com/user-attachments/assets/e25fa132-8f6a-4606-a409-2a06330b93e1" alt="Model" width="720"/>

*Figure 1. Model Architecture: Fusion of CLIP embeddings and custom classifier network.*

# Dataset
The dataset includes images classified into four groups: `Blue`, `TTR`, `Green`, and `Black`. Each image filename serves as its text description for simplicity.

- **Data Augmentation**: Applied random cropping, flipping, color jitter, and normalization to improve model robustness.
- **Train, Validation, and Test Splits**: The dataset is divided into training, validation, and testing folders for a structured evaluation of model performance.

| Dataset Split | Number of Images |
|---------------|------------------|
| Train         | _10200_          |
| Validation    | _1800_           |
| Test          | _3431_           |

*Table 1. Dataset Distribution.*

# Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/garbage-classification-clip.git
   cd garbage-classification-clip
2. **Install Required Libraries**: Ensure `torch`, `torchvision`, `clip`, `numpy`, `PIL`, `matplotlib`, and `seaborn` are installed.
   ```bash
   pip install -r requirements.txt
3.**Set Up Data Directory**:
Place your dataset in the following structure:

garbage_data/
├── CVPR_2024_dataset_Train/
│   ├── Blue/
│   ├── TTR/
│   ├── Green/
│   └── Black/
├── CVPR_2024_dataset_Val/
│   ├── Blue/
│   ├── TTR/
│   ├── Green/
│   └── Black/
└── CVPR_2024_dataset_Test/
    ├── Blue/
    ├── TTR/
    ├── Green/
    └── Black/

# Usage
1. **Train the Model**: Run the training script with the following command:
   ```bash
   python train.py

The training process will automatically save the best model as `best_model.pth` and print loss and accuracy metrics for each epoch.

2. **Test the Model**: Once training is complete, evaluate the model on the test dataset:
      ```bash
      python test.py

3.**Visualize Results**: Use the provided script to generate a confusion matrix and classification report. 
Example: 
      ```bash
      python evaluate.py

4.**Configuration**: Adjust parameters, such as batch size, learning rates, and paths, directly in the scripts for optimal results.

# Results
**Training and Validation Performance**
Metric	Training Accuracy	Validation Accuracy
| Epoch |   Training Loss   |  Validation Loss   |
|-------|-------------------|--------------------|
| 1st   |      0.5095       |      0.3256        |                      
| Final |      0.1917       |      0.2464        |

*Table 2. Training and validation accuracy across epochs.*

Test Performance
The model achieved an overall test accuracy of XX%. Below is a confusion matrix that provides insights into class-level performance.

<img src="https://github.com/user-attachments/assets/88be91ab-3650-4edb-81ce-cb4f3fba9664" alt="Screenshot 2024-11-02 215055" width="720"/>


Figure 2. Confusion Matrix on Test Data.

Classification Report: The detailed classification report is as follows:

| Class | Precision |
|---------------|------------------|
| Train         | _X_             |
| Validation    | _Y_             |
| Test          | _Z_             |


# Future Work
Potential enhancements for this project include:

**Hyperparameter Tuning**: Experiment with different optimizers and learning rates for further performance improvements.
**Model Extensions**: Test advanced transformer-based models like CLIP ViT-L for richer feature extraction.
**Additional Data**: Expand the dataset with more diverse images to improve model generalization.

# References

