import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/UNT/vk0318/Documents/Work/Code/MultiModal-DeepFake/datasets/DGM4/manipulation/HFGI/54040-HFGI.jpg')

# Define a transformation matrix for translation (simulating motion)
M = np.float32([[1, 0, 10], [0, 1, 5]])  # Translation by 10 pixels horizontally and 5 pixels vertically

# Warp the image using the transformation matrix
warped_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# Compute the difference between the original and warped images
optical_flow = cv2.absdiff(image, warped_image)

# Display the original image, warped image, and optical flow
cv2.imshow('Original Image', image)
cv2.imshow('Warped Image', warped_image)
cv2.imshow('Optical Flow', optical_flow)
cv2.waitKey(0)
cv2.destroyAllWindows()
