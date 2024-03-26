## Dental-X-RAY-Image-Detection-and-Instance-Segmentation

_**This repository leverages Detectron2, a state-of-the-art object detection and segmentation framework built on PyTorch. Detectron2 provides a flexible and efficient implementation of various algorithms, simplifying tasks like object detection, instance segmentation, and keypoint detection.**_

The Dataset contains 117 images on "Dental Radiology Scans".
![89](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation/assets/112195431/a4384ef4-4fdf-477f-a4ec-3213ac934fc4)
90% images have been taken from the dataset for Training and data-augmentation by 4 ways : 
1. +90 deg. rotation 
2. -90 deg. rotation
3. horizontal flip
4. vertical flip
5. +- 1.1 deg. rotation

and total of 269 Training images, 5 Validation images & 22 Test images have been generated from them. 

Among them, 100 images have been annotated manually, and the annotated file exported as _COCO JSON Format for Object Detection_. 

All these images used for training **Detectron2** and segmented images generated for remaining 169 images. These 169 binary segmented masks are converted into _COCO JSON Format for Object Detection_ annotation.

ALL 269 images and their combined single _.json annotation file_ used for training and 5 validation images their combined single _.json annotation file_ used for validation.

Detectron2 did **Object Detection**: Identifying and localizing objects with bounding boxes & **Semantic Segmentation**: Assigning each pixel in an image a class label on the test images.

![89_result](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation/assets/112195431/530660a6-0030-4a8b-b996-559fdeabb30c)

The Code also save the _Binary Predicted Mask_ of test set.
![89_Tooth_result](https://github.com/arpsn123/Dental-X-RAY-Image-Detection-and-Instance-Segmentation/assets/112195431/41eb0904-4a18-44c9-ab05-3a7fa37f407c)

### Integration with Roboflow and MakeSense.ai
_To streamline your object detection workflow, consider automating tasks with Roboflow and MakeSense.ai:_

**_Roboflow:_** Easily preprocess, augment, and manage your datasets to improve model performance.

**_MakeSense.ai_**: Annotate images with bounding boxes, polygons, keypoints, and more, accelerating the data labeling process.
