# Cover-Crop-Image-Segmentation
Repository containing the data, code, and results for the CoverCrop Image Segmentation project with Tensorflow and the MAsk R-CNN framework

## The objective of this project is to build a train a machine learning model to produce pixel-wise segmentation on cover crop images then detect and identify the location of the species.

image data was presegmented through an online annotator (VGG-VIA) and saved as JSON file. 
The colab notebook extracts the training and validation datasets from a zipfile in my google drive for training.
The notebook also calls the Covercrop.py file to use model with pretrained weights to conduct trasnfer learning on the covercrop dataset.
The Covercrop.py file is a modification of the coco.py codes use of the mask RCNN frmework for image segmentation found on the matterport Github Repo: https://github.com/matterport/Mask_RCNN/tree/master/samples
