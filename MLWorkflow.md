# Facial Attribute Recognition ML Workflow (CelebA, ResNet)

This readme summarizes the full workflow, machine learning models, and comparisons discussed for facial attribute recognition using the CelebA dataset. It details code structure, ML pipeline steps, and ResNet model variants.

---

## 1. Workflow Overview

This project builds a multi-label facial attribute classifier using deep learning. It leverages the CelebA dataset (over 200,000 images with 40 attributes) and implements all steps from data download to model evaluation.

---

## 2. Steps in the Workflow

### **a. Data Acquisition**
- Download CelebA from Kaggle using API credentials.
- Unzip and organize images into `/content/celeba_images/img_align_celeba/`.

### **b. Attribute File Handling**
- Load `list_attr_celeba.csv` for facial attributes.
- Index the DataFrame by image filename (`image_id`).
- Convert labels from `-1/1` to `0/1`.

### **c. Image Verification**
- Confirm image existence and display samples for sanity check.

### **d. Data Preparation**
- Split dataset (60% train, 20% val, 20% test).
- Select target attributes (e.g., Smiling, Male, Young, Wearing_Lipstick).
- Add full file paths to images.

### **e. Data Augmentation**
- Use PyTorch transforms or Keras `ImageDataGenerator` for resizing, normalization, and random augmentations.

### **f. Custom Dataset Loader**
- Implement `CelebADataset` (PyTorch) to efficiently load image-label pairs.

### **g. Model Definition**
- Use **ResNet18** (pretrained) with the final layer replaced for multi-label output (sigmoid activation).

### **h. Training & Validation**
- Train with Adam optimizer and Binary Cross Entropy loss.
- Track training/validation loss and per-attribute accuracy.

### **i. Performance Visualization**
- Plot loss curves and accuracy for monitoring.

### **j. Testing**
- Evaluate the model on the test set for final accuracy.

### **k. Model Saving**
- Save model weights for deployment or further use.

### **l. Alternate Keras/TensorFlow Pipeline**
- Show how to use Keras generators for similar data processing and augmentation.

---

## 3. Machine Learning Algorithms Used

| Model/Algorithm         | Framework      | Task                       | Details                                    |
|------------------------ |---------------|----------------------------|--------------------------------------------|
| ResNet18 (CNN)          | PyTorch       | Multi-label classification | Pretrained, final layer replaced, sigmoid  |
| ImageDataGenerator      | Keras/TensorFlow | Data augmentation         | Random transformations for image batches   |
| CelebADataset (Custom)  | PyTorch       | Data loading               | Loads image and label pairs                |
| train_test_split        | sklearn       | Data splitting             | Train/validation separation                |

---

## 4. ResNet32 vs ResNet18

**ResNet18**
- 18 layers; standard model for large images (e.g., ImageNet, facial recognition).
- ~11.7 million parameters.
- Available in PyTorch/TensorFlow/keras libraries.

**ResNet32**
- 32 layers; designed for small images (e.g., CIFAR-10: 32x32).
- ~0.5 million parameters.
- Commonly used for academic benchmarks, not standard in libraries.

| Feature             | ResNet18                  | ResNet32                 |
|---------------------|---------------------------|--------------------------|
| Layers              | 18                        | 32                       |
| Typical Input Size  | 224x224                   | 32x32                    |
| Parameters          | ~11.7 million             | ~0.5 million             |
| Use Case            | Large datasets, transfer  | Small datasets, research |
| Standard Library?   | Yes                       | No (custom)              |

---

## 5. Key Concepts

- **Multi-label classification:** Predicting several facial attributes independently.
- **Residual connections:** Used in ResNet to ease training of deep networks.
- **Data augmentation:** Improves generalization by creating diverse training samples.
- **Transfer learning:** Using pretrained CNNs for new tasks.
- **Robust data handling:** Ensures the pipeline works regardless of file format or folder structure.

---

## 6. References & Useful Links

- [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch ResNet Models](https://pytorch.org/vision/stable/models.html#id1)
- [TensorFlow ResNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)

---

## 7. Notes

- Some results in this summary refer to partial listing of repositories due to API limits.  
- For more details or full code, see source files or contact repository owner.
