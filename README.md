# shadow-segmentation
Using the publicly available SBU Dataset, semantic shadow segmentation pipeline for identifying and removing shadowed regions

Link to SBU Dataset: https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html
link to pre-trained model: https://drive.google.com/file/d/1JElEHq4b8_bmizydqPdfr3Nh-T2iCpT4/view?usp=sharing
Results on SBU Test dataset: https://drive.google.com/drive/folders/1FmTlktlyOrzbYqJtKTyQ_iC4aUQr86V6?usp=sharing

Course Project 
Course Name: Computer Vision (EE 621)

Title: Shadow Removal using deep CNNs

Abstract: Shadow detection and removal is one of the basic and challenging tasks in computer graphics and computer vision. The removal of shadow images are important pre-processing stages in computer vision and image enhancement. The presence of the shadow not only affects the visual interpretation of the image nut also the analysis of the image and the subsequent processing results.

Theory: The mapping of shadow images and their shadow-free counterparts cannot be mapped directly. However, the idea is to generate a shadow mask. A shadow mask is a monochromatic image which has the shadow area highlighted in white pixels (having pixel intensity of 255) and the remaining portion is black (having pixel intensity of 0). The complement of the shadow mask normalized by the maximum pixel intensity value (255) is known as the shadow matte. In the approach inspired from the mentioned paper, we train a convolutional autoencoder to predict the shadow matte from the input image.
