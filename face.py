import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


# Function that normalizes a vector x (i.e. |x|=1 ) #########
# > numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
#   This function is able to return one of eight different matrix norms,
#   or one of an infinite number of vector norms (described below),
#   depending on the value of the ord parameter.
#   Note: in the given functionm, U should be a vector, not a array.
#         You can write your own normalize function for normalizing
#         the colomns of an array.
def normalize(U):
    return U / LA.norm(U)


# display one face
def displayoneface(face, title=''):
    global image_count
    image_count += 1
    plt.figure(image_count)
    current_face = np.reshape(face, (64, 64), order='F')
    # image_count += 1
    # plt.figure(image_count)
    plt.title(title)
    plt.imshow(current_face, cmap=plt.cm.gray)


######### Load the data ##########

infile = open('faces.csv', 'r')
img_data = infile.read().strip().split('\n')
img = [map(int, a.strip().split(',')) for a in img_data]
pixels = []
for p in img:
    pixels += p
faces = np.reshape(pixels, (400, 4096))

######### Global Variable ##########

image_count = 0

######### Display first face #########

# Useful functions:
# > numpy.reshape(a, newshape, order='C')
#   Gives a new shape to an array without changing its data.
# > matplotlib.pyplot.figure()
# 	Creates a new figure.
# > matplotlib.pyplot.title()
#	Set a title of the current axes.
# > matplotlib.pyplot.imshow()
#	Display an image on the axes.
#	Note: You need a matplotlib.pyplot.show() at the end to display all the figures.


showindex = 0
displayoneface(faces[showindex], title='First face')

########## display a random face ###########

# Useful functions:
# > numpy.random.choice(a, size=None, replace=True, p=None)
#   Generates a random sample from a given 1-D array
# > numpy.ndarray.shape()
#   Tuple of array dimensions.
#   Note: There are two ways to order the elements in an array:
#         column-major order and row-major order. In np.reshape(),
#         you can switch the order by order='C' for row-major(default),
#         or by order='F' for column-major.

showindex = int(np.random.rand() * 400)
displayoneface(faces[showindex], title='Randomly show a face')

########## compute and display the mean face ###########

# Useful functions:
# > numpy.mean(a, axis='None', ...)
#   Compute the arithmetic mean along the specified axis.
#   Returns the average of the array elements. The average is taken over
#   the flattened array by default, otherwise over the specified axis.
#   float64 intermediate and return values are used for integer inputs.

meanface = np.mean(faces, axis=0)
displayoneface(meanface, title='Mean face')

######### substract the mean from the face images and get the centralized data matrix A ###########

# Useful functions:
# > numpy.repeat(a, repeats, axis=None)
#   Repeat elements of an array.

centerfaces = (faces - np.reshape(np.repeat(meanface, faces.shape[0], axis=0), (400, 4096))) / 255
displayoneface(centerfaces[0], title='Center face 1')

# temp show image
plt.show()

######### calculate the eigenvalues and eigenvectors of the covariance matrix #####################

# Useful functions:
# > numpy.matrix()
#   Returns a matrix from an array-like object, or from a string of data.
#   A matrix is a specialized 2-D array that retains its 2-D nature through operations.
#   It has certain special operators, such as * (matrix multiplication) and ** (matrix power).

# > numpy.matrix.transpose(*axes)
#   Returns a view of the array with axes transposed.

# > numpy.linalg.eig(a)[source]
#   Compute the eigenvalues and right eigenvectors of a square array.
#   The eigenvalues, each repeated according to its multiplicity.
#   The eigenvalues are not necessarily ordered.

vmat = np.cov(np.matrix.transpose(np.matrix(centerfaces)))
eigvals, eigvectors = LA.eig(vmat)

########## Display the first 16 principal components ##################

k = 16
sorted_indices = np.argsort(eigvals)
eigvals_top = eigvals[sorted_indices[:-k - 1:-1]]
eigvectors_top = eigvectors[:, sorted_indices[:-k - 1:-1]]

print(eigvals_top)
print(eigvectors_top)

########## Reconstruct the first face using the first two PCs #########

recontime1 = 10
reconface1 = np.zeros(4096,dtype=np.complex)
for i in range(recontime1):
    reconface1 += np.sum(centerfaces[2] * eigvectors_top[:, i]) * eigvectors_top[:, i]
reconface1 *= 255
reconface1 += meanface

displayoneface(np.real(reconface1), 'Reconstruct first')
plt.show()

########## Reconstruct random face using the first 5, 10, 25, 50, 100, 200, 300, 399  PCs ###########

#### Your Code Here ####


######### Plot proportion of variance of all the PCs ###############

# Useful functions:
# > matplotlib.pyplot.plot(*args, **kwargs)
#   Plot lines and/or markers to the Axes.
# > matplotlib.pyplot.show(*args, **kw)
#   Display a figure.
#   When running in ipython with its pylab mode,
#   display all figures and return to the ipython prompt.

#### Your Code Here ####
