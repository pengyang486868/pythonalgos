import matplotlib.pyplot as plt
import numpy as np


# display one face
def displayoneimage(image, title=''):
    global image_count
    image_count += 1
    plt.figure(image_count)
    current_face = np.reshape(image, (20, 20), order='F')
    # image_count += 1
    # plt.figure(image_count)
    plt.title(title)
    plt.imshow(current_face, cmap=plt.cm.gray)


def readdata(path, shape):
    infile = open(path, 'r')
    img_data = infile.read().strip().split('\n')
    img = [map(float, a.strip().split(',')) for a in img_data]
    pixels = []
    for p in img:
        pixels += p
    return np.reshape(pixels, shape)


hidden_dim = 25
input_dim = 400
output_dim = 10

images = readdata(r'ps5_data.csv', (-1, input_dim))
labels_raw = readdata(r'ps5_data-labels.csv', (-1, 1))

labels = np.zeros((labels_raw.shape[0], 10), dtype=int)
labels_raw_pos = 0
for lraw in labels_raw:
    labels[labels_raw_pos, int(lraw) - 1] = 1
    labels_raw_pos += 1

theta1 = readdata(r'ps5_theta1.csv', (hidden_dim + 1, -1))
theta2 = readdata(r'ps5_theta2.csv', (output_dim, -1))

image_count = 0
displayoneimage(images[1000])

plt.show()
