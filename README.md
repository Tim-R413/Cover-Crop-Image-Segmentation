# Cover Crop Image Segmentation
  Repository containing the data, code, and results for the CoverCrop Image Segmentation project with Tensorflow and the MAsk R-CNN framework

  The objective of this project is to build and train a machine learning model to produce pixel-wise segmentation on cover crop images then detect and identify the location of the species.

  Image data was presegmented through an online annotator (VGG-VIA) and saved as JSON file. The colab notebook extracts the training and validation datasets from a zipfile in my google drive for training.The notebook also calls the Covercrop.py file to use model with pretrained weights to conduct trasnfer learning on the covercrop dataset. The Covercrop.py file is a modification of the coco.py codes use of the mask RCNN frmework for image segmentation found on the matterport Github Repo: https://github.com/matterport/Mask_RCNN/tree/master/samples


## Preliminary scope and project objectives
   the intial course of action for this project will be to explore different approaches to Semantic Segmentation. The first bein a region based approach using the Mask RCNN framework from the Matterport Repo. The second will also follow a region based approach using the UNet framework similar to the example that Tensorflow provided. The final notebook will be an attempt to use a Fully Convolutional Network based approach. The progress made with each approach as well as further detatils will be outlined in their respective folders in this repo. 
   
## UPDATE:
there are some complications with displaying/ uploading the colab notebooks to the Github sub directories, torubleshooting the issue currently. 
