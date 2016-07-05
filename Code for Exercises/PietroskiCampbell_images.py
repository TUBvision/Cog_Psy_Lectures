# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 14:21:51 2016

@author: Will
"""
import cv2 
import matplotlib.pyplot as plt
import numpy as np

# Import images
face1=cv2.imread('C:\Users\Will\Documents\gitrepos\Cog_Psy_Lectures\Figures\Hybrid_Images\monroe2.jpg')
face2=cv2.imread('C:\Users\Will\Documents\gitrepos\Cog_Psy_Lectures\Figures\Hybrid_Images\einstein.jpg')

# Select only single colour channel
face1=face1[:,:,0]
face2=face2[:,:,0]

plt.figure(1)
plt.subplot(2,4,1)
plt.imshow(face1,cmap='gray')
plt.title('Input Images')
plt.axis('off')
plt.subplot(2,4,5)
plt.imshow(face2,cmap='gray')
plt.axis('off')

# Fourier transform
ftFace1=np.fft.fftn(np.double(face1))
ftFace2=np.fft.fftn(np.double(face2))

plt.subplot(2,4,2)
plt.imshow(np.log(np.abs(ftFace1)),cmap='gray')
plt.title('FFT Image Magnitude')
plt.axis('off')
plt.subplot(2,4,6)
plt.imshow(np.log(np.abs(ftFace2)),cmap='gray')
plt.axis('off')
plt.subplot(2,4,3)
plt.imshow(np.angle(ftFace1),cmap='gray')
plt.title('FFT Image Phase')
plt.axis('off')
plt.subplot(2,4,7)
plt.imshow(np.angle(ftFace2),cmap='gray')
plt.axis('off')


# Reconstruction with phase from other image
amp1Phase2=np.fft.ifftn(np.abs(ftFace1)*np.cos(np.angle(ftFace2)) + 0j*np.abs(ftFace1)*np.sin(np.angle(ftFace2)))
amp2Phase1=np.fft.ifftn(np.abs(ftFace2)*np.cos(np.angle(ftFace1)) + 0j*np.abs(ftFace2)*np.sin(np.angle(ftFace1)))

plt.subplot(2,4,4)
plt.imshow(np.real(amp1Phase2),cmap='gray')
plt.title('Recontruction with Phase from other Image')
plt.axis('off')
plt.subplot(2,4,8)
plt.imshow(np.real(amp2Phase1),cmap='gray')
plt.axis('off')

