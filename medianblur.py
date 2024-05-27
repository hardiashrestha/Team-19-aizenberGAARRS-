import cv2

import numpy as np

import matplotlib.pyplot as plt
# Reading the image
img = cv2.imread(r'D:/dat/ISIC_2989732.jpg')
# Displaying the original image
img = cv2.resize(img, [256,256])
plt.imshow(img)
img = cv2.medianBlur(img, 9)

# Displaying the blurred image
plt.imshow(img)
plt.show()
