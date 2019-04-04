from PIL import Image
import numpy as np
import math
from scipy import signal
import os

#1,1
#returns a box filter of size nxn -- a numpy array
def boxfilter(n):

    #if even, throw error
    if (n%2 == 0):
        try:
            assert False, "Dimension must be odd"
        except AssertionError, e:
            raise Exception(e.args)
    #else make little box
    else:
        print(np.full((n, n), 0.4))

#boxfilter(3)
#boxfilter(4)
#boxfilter(5)

#1,2
def gauss1d(sigma):
    #sigma*6
    len_0 = (sigma*6)

    #round up
    len_1 = math.ceil(len_0)

    #...to the next odd int.
    if (len_1%2 == 0):
        len_2 = len_1 + 1
    else: len_2 = len_1

    #basic distance from center array
    array_1 = np.arange(-((len_2-1)/2)-1, (len_2-1)/2)+1

    #exp(- x^2 / (2*sigma^2))
    array_2 = np.exp(-np.power(array_1,2)/(2*(sigma**2)))

    #normalizing
    sum_0 = np.sum(array_2)
    final_array = array_2/sum_0
    #print(final_array)
    return final_array

#gauss1d(0.3)
#gauss1d(0.5)
#gauss1d(1)
#gauss1d(2)


#1,3
def gauss2d(sigma):
    #getting the 1d array
    one_d_array = gauss1d(sigma)
    #getting 2d array from that
    two_d_array_1 = one_d_array[np.newaxis]

    #getting it's transpose
    transpose_array = np.transpose(two_d_array_1)

    #convolving that
    final_array = signal.convolve2d(two_d_array_1,transpose_array)
    #print(final_array)
    #print(final_array)
    return final_array


#gauss2d(0.5)
#gauss2d(1)

#1,4,a
def gaussconvolve2d(array,sigma):
    #making a filter from the above function
    filter_1 = gauss2d(sigma)
    #convolving that
    final_array = signal.convolve2d(array, filter_1, 'same')
    #print(final_array)
    return final_array

#TODO: this explanation in the other file!
#1,4,b
#cwd = os.getcwd()
#print(cwd)

#opening image of dog
image_dog = Image.open('doggo.jpg')

#convert it to a greyscale
image_greyscale = image_dog.convert('L')

#convert to numpy
image_numpy = np.asarray(image_greyscale)

#convert to double array format
imagArr = np.asfarray(image_numpy)
#print(imagArr)

# Numpy array and run
conversion_image = gaussconvolve2d(imagArr,3)
#print(conversion_image)

#covert the array back to unsigned integer format for storage and display
final_image = Image.fromarray(conversion_image.astype('uint8'))
final_image.save('final_doggo.png', 'PNG')

#1,4,c
#Use PIL to show both the original and filtered images.

#1,5
#TODO: this explanation in the other file!

#2,1
#open image of dog
image_dog_1 = Image.open('doggo.jpg')

width, height = image_dog_1.size
#convert to numpy
image_dog_numpy = np.asarray(image_dog_1)
#convert to double array format
imagDogArr = np.asarray(image_dog_numpy)

#new final array for final pic
conversion_dog_array = np.zeros((height, width, 3))

#new assignments after convolutions
conversion_dog_array[:, :, 0] = gaussconvolve2d(imagDogArr[:, :, 0],3)
conversion_dog_array[:, :, 1] = gaussconvolve2d(imagDogArr[:, :, 1],3)
conversion_dog_array[:, :, 2] = gaussconvolve2d(imagDogArr[:, :, 2],3)

final_image_dog = Image.fromarray(conversion_dog_array.astype('uint8'))
final_image_dog.save('final_split_dog.png', 'PNG')

#2,2
#open image of cat
image_cat = Image.open('cat.jpg')

width_cat, height_cat = image_cat.size
#convert to numpy
image_cat_numpy = np.asarray(image_cat)
#convert to double array format
imagCatArr = np.asarray(image_cat_numpy)

#new array for intermediary blurry cat
conversion_cat_array = np.zeros((height_cat, width_cat, 3))

#new assignments after convolutions
conversion_cat_array[:, :, 0] = gaussconvolve2d(imagCatArr[:, :, 0],3)
conversion_cat_array[:, :, 1] = gaussconvolve2d(imagCatArr[:, :, 1],3)
conversion_cat_array[:, :, 2] = gaussconvolve2d(imagCatArr[:, :, 2],3)

foggy_image_cat = Image.fromarray(conversion_cat_array.astype('uint8'))

#new final array for final pic
final_sharp_cat = np.zeros((height_cat, width_cat, 3))
#subtracting blurry image
final_sharp_cat = (imagCatArr - foggy_image_cat) #+ 128
final_sharp_cat = Image.fromarray(final_sharp_cat.astype('uint8'))
final_sharp_cat.save('final_sharp_cat.png', 'PNG')

#2,3
#new array for hybrid image
hybrid_array = np.zeros((height, width, 3))

#have to make this an array again
final_sharp_cat = np.asarray(final_sharp_cat)

#have to make this an array again
final_image_dog = np.asarray(final_image_dog)

#add the sharpened image to the blurred image
hybrid_array[:, :, 0] = final_sharp_cat[:, :, 0] + final_image_dog[:, :, 0]
hybrid_array[:, :, 1] = final_sharp_cat[:, :, 1] + final_image_dog[:, :, 1]
hybrid_array[:, :, 2] = final_sharp_cat[:, :, 2] + final_image_dog[:, :, 2]

#clamping
final_hybrid = np.zeros(hybrid_array.shape)
final_hybrid[:, :, 0] = np.clip(hybrid_array, 0, 255)
final_hybrid[:, :, 1] = np.clip(hybrid_array, 0, 255)
final_hybrid[:, :, 2] = np.clip(hybrid_array, 0, 255)

final_final_hybrid = Image.fromarray(final_hybrid.astype('uint8'))
final_final_hybrid.save('final_hybrid.png', 'PNG')

















