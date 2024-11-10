## Dental-X-RAY-Image-Detection-and-Instance-Segmentation

<!-- Repository Overview Badges -->
<div align="center">
    <img src="https://img.shields.io/github/stars/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=ffca28" alt="GitHub Repo Stars">
    <img src="https://img.shields.io/github/forks/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=00aaff" alt="GitHub Forks">
    <img src="https://img.shields.io/github/watchers/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=00e676" alt="GitHub Watchers">
</div>

<!-- Issue & Pull Request Badges -->
<div align="center">
    <img src="https://img.shields.io/github/issues/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=ea4335" alt="GitHub Issues">
    <img src="https://img.shields.io/github/issues-pr/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=ff9100" alt="GitHub Pull Requests">
</div>

<!-- Repository Activity & Stats Badges -->
<div align="center">
    <img src="https://img.shields.io/github/last-commit/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=673ab7" alt="GitHub Last Commit">
    <img src="https://img.shields.io/github/contributors/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=388e3c" alt="GitHub Contributors">
    <img src="https://img.shields.io/github/repo-size/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=303f9f" alt="GitHub Repo Size">
</div>

<!-- Language & Code Style Badges -->
<div align="center">
    <img src="https://img.shields.io/github/languages/count/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=607d8b" alt="GitHub Language Count">
    <img src="https://img.shields.io/github/languages/top/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation?style=for-the-badge&logo=github&logoColor=white&color=4caf50" alt="GitHub Top Language">
</div>

<!-- Maintenance Status Badge -->
<div align="center">
    <img src="https://img.shields.io/badge/Maintenance-%20Active-brightgreen?style=for-the-badge&logo=github&logoColor=white" alt="Maintenance Status">
</div>



_**This repository leverages Detectron2, a state-of-the-art object detection and segmentation framework built on PyTorch. Detectron2 provides a flexible and efficient implementation of various algorithms, simplifying tasks like object detection, instance segmentation, and keypoint detection.**_

---

### Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Augmentation](#data-augmentation)
3. [Annotation Process](#annotation-process)
4. [Model Training](#model-training)
5. [Results and Output](#results-and-output)
6. [Integration with Roboflow and MakeSense.ai](#integration-with-roboflow-and-makesenseai)
7. [Setup Instructions](#setup-instructions)
8. [Tech Stack](#tech-stack)
9. [Conclusion](#conclusion)

---

### Dataset Overview

The dataset consists of **117 images** focused on "Dental Radiology Scans," which serve as the foundation for training and evaluating detection and segmentation algorithms in the dental field. This dataset is critical for developing models that can accurately interpret dental images and assist in diagnosis.

![Sample Dental X-Ray Image](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation/assets/112195431/a4384ef4-4fdf-477f-a4ec-3213ac934fc4)

### Data Augmentation

To enhance the model's robustness and generalization, **90% of the images** were utilized for training. Data augmentation techniques were employed to generate a total of **269 training images**, improving the dataset's diversity. The following transformations were applied:

1. **+90° Rotation**: Rotating the images clockwise by 90 degrees to simulate different orientations.
2. **-90° Rotation**: Rotating the images counterclockwise by 90 degrees to account for variations in imaging angles.
3. **Horizontal Flip**: Flipping the images horizontally, mirroring the content to simulate lateral perspectives.
4. **Vertical Flip**: Flipping the images vertically, creating a top-down mirror effect to enrich the training dataset.
5. **±1.1° Rotation**: Slight random rotations to simulate variations in the angle of image capture, improving the model's ability to generalize.

After applying these augmentations, the dataset composition includes:
- **269 Training Images**
- **5 Validation Images**
- **22 Test Images**

### Annotation Process

Out of the total images, **100 images** were manually annotated using the **COCO JSON Format** for object detection. The COCO format is a widely adopted standard for object detection tasks, allowing for efficient integration with various machine learning frameworks, including Detectron2. This meticulous annotation process ensures high-quality training data, critical for achieving robust model performance.

### Model Training

All **269 images** and their combined single **.json annotation file** were utilized for training the model. The training involved using **Detectron2**, which executed both:

- **Object Detection**: Identifying and localizing objects within the images by drawing bounding boxes around detected items. This functionality is essential for locating dental features accurately.
- **Semantic Segmentation**: Assigning a class label to each pixel in an image, enabling detailed localization of dental structures. This is crucial for identifying different dental elements in a radiology scan.

![Object Detection Result](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation/assets/112195431/530660a6-0030-4a8b-b996-559fdeabb30c)

### Results and Output

The code saves the **Binary Predicted Mask** of the test set, providing clear visualizations of the model's performance on unseen data. These binary masks highlight the areas of interest, allowing for quick assessment of model accuracy.

![Binary Predicted Mask Example](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation/assets/112195431/41eb0904-4a18-44c9-ab05-3a7fa37f407c)

### Integration with Roboflow and MakeSense.ai

To streamline your object detection workflow, consider automating tasks with **Roboflow** and **MakeSense.ai**:

- **Roboflow**: This platform simplifies dataset management by offering tools to preprocess, augment, and version control datasets. By integrating Roboflow, you can significantly enhance model performance and facilitate collaboration.

- **MakeSense.ai**: A powerful annotation tool that allows you to annotate images with bounding boxes, polygons, keypoints, and more. This tool accelerates the data labeling process, enabling you to create high-quality annotations quickly and efficiently, which is essential for training robust models.

---

### Setup Instructions

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation.git
   cd Dental-X-RAY-Image-Detection-and-Instance-Segmentation
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Make sure you have Python 3.7 or later installed. Use the following command to install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Detectron2**:
   Follow the [Detectron2 installation instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to ensure it is properly set up in your environment.

5. **Download the Dataset**:
   Ensure that you have the dataset stored in the correct directory structure as required by the code.

6. **Run the Training Script**:
   Execute the training script to begin the training process:
   ```bash
   python train.py
   ```

7. **Evaluate the Model**:
   After training, run the evaluation script to assess the model's performance:
   ```bash
   python evaluate.py
   ```

### Tech Stack
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Detectron2](https://img.shields.io/badge/Detectron2-v0.6.1-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9.0-red)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.5.1-blue)
![NumPy](https://img.shields.io/badge/NumPy-v1.20.2-orange)
![COCO JSON Format](https://img.shields.io/badge/COCO%20JSON%20Format-v2.0.0-yellow)

- **Python**: The primary programming language for developing the model and handling data manipulation.
- **Detectron2**: A powerful object detection and segmentation framework that simplifies the process of building and training models.
- **PyTorch**: The deep learning framework that underpins Detectron2, providing efficient tensor computation and dynamic neural network capabilities.
- **OpenCV**: A library used for image processing tasks such as image manipulation, augmentation, and visualization.
- **NumPy**: A fundamental package for scientific computing with Python, used for handling arrays and mathematical operations.
- **COCO JSON Format**: A standard for representing object detection data, allowing for easy integration with machine learning frameworks.

---

### Conclusion

This project exemplifies the application of cutting-edge machine learning techniques to the field of dental radiology, enhancing the accuracy and efficiency of dental image analysis through advanced object detection and segmentation methodologies.
