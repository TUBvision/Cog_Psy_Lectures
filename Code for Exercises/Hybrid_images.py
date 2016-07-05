# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Import images
image1=cv2.imread('C:\Users\Will\Downloads\monroe2.jpg')
image2=cv2.imread('C:\Users\Will\Downloads\einstein.jpg')

# Resize images
image1=cv2.resize(image1, (0,0), fx=0.5, fy=0.5) 
image2=cv2.resize(image2, (0,0), fx=0.5, fy=0.5) 


# Use Guassian filter function as previously made
def Gaussian_filter(gamma, N):
    """
    Create the Gaussian filters
    """
    Amp = 1 / (2*np.pi*gamma**2)
    x = np.arange(-N,N,0.5)
    y = np.arange(-N,N,0.5)
    [x,y] = np.meshgrid(x,y)
    g = Amp * np.exp (-(x**2 + y **2)/(2*gamma**2))
    return g
    
# Gaussian parameters
s1=4
s2=5
N=20

# Filtering and image extraction
low_freq=np.zeros(image1.shape)
high_freq=np.zeros(image1.shape)
for k in range(3):
    low_freq[:,:,k]=convolve2d(image2[:,:,k],Gaussian_filter(s1,N),mode="same")
    high_freq[:,:,k]=image1[:,:,k] - convolve2d(image1[:,:,k],Gaussian_filter(s2,N),mode="same")


# Plot images
plt.figure(1,figsize=(30,10))
plt.subplot(1,3,1)
plt.imshow(low_freq[:,:,0]+high_freq[:,:,0],cmap='gray')
plt.subplot(1,3,2)
plt.imshow(low_freq[:,:,0], cmap='gray')
plt.subplot(1,3,3)
plt.imshow(high_freq[:,:,0],cmap='gray')
