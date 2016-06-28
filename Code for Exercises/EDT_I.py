# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:07:03 2016

@author: will
"""

"""
TO BE GIVEN
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

"""
Question 1 - Guassian Blur
"""
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

im = Image.open("/home/will/Downloads/lenaTest3.jpg")
arr = np.array(im)

sigma=5
N=5
gaus = Gaussian_filter(sigma,5)

plt.figure(1)
plt.subplot(1,4,1)
plt.imshow(arr,cmap='gray')
plt.subplot(1,4,2)
plt.imshow(gaus,cmap='gray')
plt.subplot(1,4,3)
plt.imshow(convolve2d(arr,gaus,mode='same'),cmap='gray')#,vmin=0,vmax=255)
plt.subplot(1,4,4)  
plt.imshow(arr-convolve2d(arr,gaus,mode='same'),cmap='gray')#,vmin=0,vmax=255)
   
   
"""
Question 2 - DOG
"""
gaus1 = Gaussian_filter(1,1)
gaus2 = Gaussian_filter(2,2)
plt.figure(2)
plt.imshow(convolve2d(arr,gaus2,mode='same')-convolve2d(arr,gaus1,mode='same'), cmap='gray')


"""
 Question 3 - ODOG
"""
def Oriented_gaussian_filter(a,b,c,d,gamma, N):
    """
    Create the Gaussian filters
    [a,b]->[x,y] 1st filter orientation vector
    [c,d]->[x,y] 2nd filter orientation vector
    """
    Amp = 1 / (2*np.pi*(gamma**2))
    x = np.arange(-N,N,0.5)
    y = np.arange(-N,N,0.5)
    [x,y] = np.meshgrid(x,y)
    A=(a*x + b*y)**2
    B=(c*x + d*y)**2
    g = Amp * np.exp((-A/(2*gamma**2))-(-B/(2*gamma**2)))
    return g

# horizontal DOG filter
OG1=Oriented_gaussian_filter(2,2,0,0,10,20)


plt.figure(3)
plt.subplot(2,1,1)
plt.imshow(OG1,cmap='gray')
plt.subplot(2,1,2)
plt.imshow(convolve2d(arr,OG1,mode='same'),cmap='gray')

OG2=Oriented_gaussian_filter(1,0,0,1,2,5)
ODOGh=convolve2d(arr,OG2,mode='same')-convolve2d(arr,OG1,mode='same')

# vertical DOG filter
OG3=Oriented_gaussian_filter(0,1,1,0,1,5)
OG4=Oriented_gaussian_filter(0,1,1,0,2,5)
ODOGv=convolve2d(arr,OG3,mode='same')-convolve2d(arr,OG4,mode='same')

plt.figure(4)
plt.subplot(2,2,1)
plt.imshow(ODOGh,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(ODOGv,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(OG1,cmap='gray')
plt.subplot(2,2,4)
plt.imshow(OG3,cmap='gray')


"""
Question 4 - Derivative Gaussians
"""

def Diff_oriented_gaussian_filter(gamma, N):
    """
    Create a derivative Gaussian filters
    """
    Amp = 1 / (2*np.pi*(gamma**2))
    x = np.arange(-N,N,0.5)
    y = np.arange(-N,N,0.5)
    [x,y] = np.meshgrid(x,y)
    A=(x + y)**2
    one = (Amp*2*x) * np.exp(-A/(2*gamma**2))
    two = (Amp*2*y) * np.exp(-A/(2*gamma**2))
    return one-two

dog1=Diff_oriented_gaussian_filter(5, 10)

plt.figure(5)
plt.subplot(2,1,1)
plt.imshow(dog1,cmap='gray')
plt.subplot(2,1,2)
plt.imshow(convolve2d(arr,OG1,mode='same'),cmap='gray')
"""
Question 1 - Edge Detection 2 -  Image contrast gradients
"""
grad=np.gradient(arr)
mag_grad=np.sqrt((grad[0]**2)+(grad[0]**2))

plt.figure(4)
plt.subplot(1,3,1)
plt.imshow(arr,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(grad[0]+grad[1],cmap='gray')
plt.subplot(1,3,3)
plt.imshow(mag_grad,cmap='gray')

 