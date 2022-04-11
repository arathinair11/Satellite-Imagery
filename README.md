## Mapping Deprived Areas In Low & Middle Income Countries Using Satellite Images

## About
This repository consist of Python code for extracting cloud-free Sentinel-2 images from Google Earth Engine and implementation of CNN & MLP Autoencoder for image reconstruction and training image classification usng Pytorch.

#### Dependencies
- Python3, Scikit-learn
- Pytorch, PIL
```
- pip install earthengine-api
- pip install folium
```

## Files 

#### **1. Cloud Free Sentinel-2 Image Extraction**
- Create a [Google Earth Engine](https://earthengine.google.com) account before running the code.

  - [Lagos_Cloud_Free_Satellite_Image.ipynb](https://github.com/arathinair11/Satellite-Imagery/blob/main/Lagos/Lagos_Cloud_Free_Satellite_Image.ipynb) :  Code for extracting cloud free Sentinel-2 image for Lagos in TIF format
  - [Accra_Cloud_Free_Satellite_Image.ipynb](https://github.com/arathinair11/Satellite-Imagery/blob/main/Accra/Accra_Cloud_Free_Satellite_Image.ipynb) : Code for extracting cloud free Sentinel-2 image for Accra in TIF format
  - [Nairobi_Cloud_Free_Satellite_Image.ipynb](https://github.com/arathinair11/Satellite-Imagery/blob/main/Nairobi/Nairobi_Cloud_Free_Satellite_Image.ipynb) : Code for extracting cloud free Sentinel-2 image for Nairobi in TIF format
   - Parameters chosen for image fine-tuning : *START_DATE,END_DATE,CLOUD_FILTER,CLD_PRB_THRESH,CLD_PRJ_DIST,BUFFER*
- [Image_Extraction.py](https://github.com/arathinair11/Satellite-Imagery/blob/main/Image%20Clipping%20%26%20PNG%20Conversion/Image_Extraction.py) : Clipping 10 x 10 pixels from TIF files and filtering PNG images 
- [ConvertToPNG.py](https://github.com/arathinair11/Satellite-Imagery/blob/main/Image%20Clipping%20%26%20PNG%20Conversion/ConvertToPNG.py) : Converting 10 x 10 pixel TIF images to PNG format
- [TIF Files](https://drive.google.com/drive/folders/1y-t8iV_hT73FOQrflBfAui3L1wc6osST?usp=sharing) : Download final Cloud-Free TIF images for Lagos, Accra and Nairobi

#### **2. Image Reconstruction**
- [CNN Autoencoder](https://github.com/arathinair11/Satellite-Imagery/blob/main/Autoencoder/CNN_Autoencoder.py) : Code for image reconstruction using Convolutional Autoencoder
- [MLP Autoencoder](https://github.com/arathinair11/Satellite-Imagery/blob/main/Autoencoder/MLP_Autoencoder.py) : Code for image reconstruction using  Multilayer Perceptron Autoencoder

#### **3. Image Classification** 
- [Image_Classification.py](https://github.com/arathinair11/Satellite-Imagery/blob/main/Autoencoder/Image_Classfication.py) : Classification of labeled images in the training dataset using the saved [Encoder model](https://github.com/arathinair11/Satellite-Imagery/blob/main/Autoencoder/Path/enoder_autoencoder.pth) from the Autoencoder


