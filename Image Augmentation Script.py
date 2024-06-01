import os
import random
import cv2
import numpy as np
from skimage.util import random_noise

def translate_image(image, max_translation):
    rows, cols = image.shape[:2]
    tx = int(random.uniform(-max_translation, max_translation) * cols)
    ty = int(random.uniform(-max_translation, max_translation) * rows)

    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return translated_image

def scale_image(image, scale_range):
    rows, cols = image.shape[:2]
    scale = random.uniform(*scale_range)

    scaled_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    if scale < 1.0:
        # Pad the image to original size
        pad_y = (rows - scaled_image.shape[0]) // 2
        pad_x = (cols - scaled_image.shape[1]) // 2
        scaled_image = cv2.copyMakeBorder(scaled_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT)
    else:
        # Crop the image to original size
        start_x = (scaled_image.shape[1] - cols) // 2
        start_y = (scaled_image.shape[0] - rows) // 2
        scaled_image = scaled_image[start_y:start_y + rows, start_x:start_x + cols]

    return scaled_image

def rotate_image(image, max_angle):
    rows, cols = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return rotated_image

def flip_image(image):
    flip_code = random.choice([-1, 0, 1])  # Randomly choose to flip horizontally, vertically, or both
    flipped_image = cv2.flip(image, flip_code)
    return flipped_image

def add_noise(image, mode='gaussian'):
    noisy_img = random_noise(image, mode=mode)
    noisy_img = np.array(255 * noisy_img, dtype='uint8')
    return noisy_img

def augment_image(image):
    # Apply random translation
    image = translate_image(image, max_translation=0.1)  # max translation as 20% of image dimensions

    # Apply random scaling
    image = scale_image(image, scale_range=(0.9, 1.05))  # scaling range between 80% and 120%

    # Apply random rotation
    image = rotate_image(image, max_angle=30)  # maximum rotation angle of 30 degrees

    # Apply random flip
    image = flip_image(image)

    # Add random noise
    image = add_noise(image, mode='gaussian')

    return image

def augment_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(file_path)
            if image is not None:
                # Augment the image
                augmented_image = augment_image(image)

                # Save the augmented image with a prefix 'aug_' in the same folder
                augmented_image_path = os.path.join(folder_path, f'aug_{filename}')
                cv2.imwrite(augmented_image_path, augmented_image)
                print(f'Saved augmented image: {augmented_image_path}')

# Example usage
folder_path = '' 
augment_images_in_folder(folder_path)
