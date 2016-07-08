# -*- coding: utf-8 -*-
"""
KOGW-PM-KNP: Edge Detection Tutorial Supplementary Code

Please note: The convolution function convolve2d is quite computationally 
expensive and so you may need a little patience whilst it processes.
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

"""
Task 1 - Guassian Filtering of an Image

"""
def Gaussian_filter(sigma, N):
    """
    This function uses the 2D Gaussian equation to create a filter of size [N,N].
    Applying these filters to images is often called Gaussian blurring.
    
    Parameters
    ----------
    sigma : integer value - standard deviation of the Gaussian distribution
    N     : integer value - size of filter
    
    Returns
    ----------
    g : 2D Gaussian Filter 
    """
    
    Amp = 1 / (2*np.pi*sigma**2)
    x = np.arange(-N,N,0.5)
    y = np.arange(-N,N,0.5)
    [x,y] = np.meshgrid(x,y)
    g = Amp * np.exp (-(x**2 + y **2)/(2*sigma**2))
    return g

# Import image
im = Image.open("/home/will/gitrepos/Cogntive_Psychology_Lectures/Figures/Edge_detection_1/lena.jpg")
im = np.array(im)

# Define parameters and create filter
sigma=5
N=5
gaus = Gaussian_filter(sigma,N)

# Image-filter convolution.
blurred_image = convolve2d(im,gaus,mode='same')

# Question 1
plt.figure(1)
plt.subplot(1,3,1)
plt.imshow(im,cmap='gray')
plt.title('Input image')
plt.subplot(1,3,2)
plt.imshow(gaus,cmap='gray')
plt.title('Gaussian filter')
plt.subplot(1,3,3)
plt.imshow(blurred_image,cmap='gray')
plt.title('Output image')


# Question 2
plt.figure(2)
plt.subplot(1,3,1)
plt.imshow(Gaussian_filter(1,5),cmap='gray')
plt.title('sigma = 1')
plt.subplot(1,3,2)
plt.imshow(Gaussian_filter(2,5),cmap='gray')
plt.title('sigma = 2')
plt.subplot(1,3,3)
plt.imshow(Gaussian_filter(5,5),cmap='gray')
plt.title('sigma = 5')


# Question 3
low_freq_image = convolve2d(im,gaus,mode='same')
"""
high_freq_image = ?????
plt.figure(3)
plt.imshow(high_freq_image,cmap='gray')
""" 
   
"""
Task 2 - Difference of Gaussian filtering of an image

"""
gaus1 = Gaussian_filter(5,5)
gaus2 = Gaussian_filter(10,10)
blur1 = convolve2d(im,gaus2,mode='same')
blur2 = convolve2d(im,gaus1,mode='same')

plt.figure(4)
plt.imshow(blur1 - blur2, cmap='gray')


"""
Task 3 - Hybrid Images

"""

 # Import images
monroe=cv2.imread('/home/will/gitrepos/Cogntive_Psychology_Lectures/Figures/Edge_detection_1/monroe.jpg')
einstein=cv2.imread('/home/will/gitrepos/Cogntive_Psychology_Lectures/Figures/Edge_detection_1/einstein.jpg')

# Resize images
monroe=cv2.resize(monroe, (0,0), fx=0.5, fy=0.5) 
einstein=cv2.resize(einstein, (0,0), fx=0.5, fy=0.5) 

# Take single colour channel
monroe=monroe[:,:,0]
einstein=einstein[:,:,0]

# Gaussian parameters
s1=4
s2=5
N=20

# Filtering and image extraction
low_freq_einstein=convolve2d(einstein,Gaussian_filter(s1,N),mode="same")
high_freq_monroe=monroe - convolve2d(monroe,Gaussian_filter(s2,N),mode="same")


# Plot images
plt.figure(1,figsize=(30,10))
plt.subplot(1,3,1)
plt.imshow(low_freq_einstein+high_freq_monroe,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(low_freq_einstein, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(high_freq_monroe,cmap='gray')


"""
Task 4 - Phase and Fourier Transforms
"""

# Fourier transform
ftmonroe=np.fft.fftn(np.double(monroe))
fteinstein=np.fft.fftn(np.double(einstein))

# Amplitude of Images
ampmonroe = np.log(np.abs(ftmonroe))
ampeinstein = np.log(np.abs(fteinstein))

# Phase of Images
phasemonroe = np.angle(ftmonroe)
phaseeinstein = np.angle(fteinstein)

# Reconstruction with phase from other image
"""
amp1Phase2=???
amp2Phase1=???
"""
plt.figure(1)
plt.subplot(2,4,1)
plt.imshow(monroe,cmap='gray')
plt.title('Input Images')
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(einstein,cmap='gray')
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(ampmonroe,cmap='gray')
plt.title('FFT Image Magnitude')
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(ampeinstein,cmap='gray')
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(phasemonroe,cmap='gray')
plt.title('FFT Image Phase')
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(phaseeinstein,cmap='gray')
plt.axis('off')
#plt.subplot(2,4,4)
#plt.imshow(np.real(amp1Phase2),cmap='gray')
#plt.title('Recontruction with Phase from other Image')
#plt.axis('off')
#plt.subplot(2,4,8)
#plt.imshow(np.real(amp2Phase1),cmap='gray')
#plt.axis('off')

