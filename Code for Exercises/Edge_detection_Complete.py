# -*- coding: utf-8 -*-
"""
KOGW-PM-KNP: Edge Detection Tutorial Supplementary Code with Answers

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
im = Image.open("lena.jpg")
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
G1=Gaussian_filter(1,5)
G2=Gaussian_filter(2,5)
G5=Gaussian_filter(5,5)

plt.figure(2)
plt.subplot(2,3,1)
plt.imshow(convolve2d(im,G1,mode='same'),cmap='gray')
plt.title('sigma = 1')
plt.subplot(2,3,2)
plt.imshow(convolve2d(im,G2,mode='same'),cmap='gray')
plt.title('sigma = 2')
plt.subplot(2,3,3)
plt.imshow(convolve2d(im,G5,mode='same'),cmap='gray')
plt.title('sigma = 5')
plt.subplot(2,3,4)
plt.imshow(G1,cmap='gray')
plt.subplot(2,3,5)
plt.imshow(G2,cmap='gray')
plt.subplot(2,3,6)
plt.imshow(G5,cmap='gray')


# Question 3
low_freq_image = convolve2d(im,gaus,mode='same')
high_freq_image = im - low_freq_image

plt.figure(3)
plt.imshow(high_freq_image,cmap='gray')

   
"""
Task 2 - Difference of Gaussian filtering of an image

"""
# Optimum sigma and N values below
gaus1 = Gaussian_filter(1,10)
gaus2 = Gaussian_filter(2,10)
DOG = gaus1-gaus2
output = convolve2d(im,DOG,mode='same')

plt.figure(4)
plt.imshow(output, cmap='gray')


# Answer to T2.3
blur1 = convolve2d(im,gaus2,mode='same')
blur2 = convolve2d(im,gaus1,mode='same')
plt.figure(5)
plt.subplot(2,1,1)
plt.imshow(blur2-blur1, cmap='gray')
plt.subplot(2,1,2)
plt.imshow(output,cmap='gray')

"""
Task 3 - Hybrid Images

"""

# Import images / Resize image / Change data type / Select single color channel from RGB
monroe=Image.open('monroe.jpg')
einstein=Image.open('einstein.jpg')
monroe=monroe.resize([300,400],1)
einstein=einstein.resize([300,400],1)
einstein = np.array(einstein)
monroe = np.array(monroe)
einstein=einstein[:,:,0]
monroe=monroe[:,:,0]

# Gaussian parameters
s1=4
s2=6
N=10

# Filtering and image extraction
low_freq_einstein=convolve2d(einstein,Gaussian_filter(s1,N),mode="same")
high_freq_monroe=monroe - convolve2d(monroe,Gaussian_filter(s2,N),mode="same")


# Plot images
plt.figure(5,figsize=(30,10))
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

# Magnitude of Images
magmonroe = np.log(np.abs(ftmonroe))
mageinstein = np.log(np.abs(fteinstein))

# Phase of Images
phasemonroe = np.angle(ftmonroe)
phaseeinstein = np.angle(fteinstein)

# Reconstruction with phase from other image
amp1Phase2=np.fft.ifftn(np.abs(ftmonroe)*np.cos(np.angle(fteinstein)) + 0j*np.abs(ftmonroe)*np.sin(np.angle(fteinstein)))
amp2Phase1=np.fft.ifftn(np.abs(fteinstein)*np.cos(np.angle(ftmonroe)) + 0j*np.abs(fteinstein)*np.sin(np.angle(ftmonroe)))


plt.figure(1)
plt.subplot(2,4,1)
plt.imshow(monroe,cmap='gray')
plt.title('Input Images')
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(einstein,cmap='gray')
plt.axis('off')
plt.subplot(2,4,2)
plt.imshow(magmonroe,cmap='gray')
plt.title('FFT Image Magnitude')
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(mageinstein,cmap='gray')
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(phasemonroe,cmap='gray')
plt.title('FFT Image Phase')
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(phaseeinstein,cmap='gray')
plt.axis('off')
plt.subplot(2,4,4)
plt.imshow(np.real(amp1Phase2),cmap='gray')
plt.title('Recontruction with Phase from other Image')
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(np.real(amp2Phase1),cmap='gray')
plt.axis('off')

