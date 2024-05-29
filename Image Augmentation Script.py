from PIL import Image, ImageEnhance
import numpy as np
import random

def translate_image(image, max_translate=(10, 10)):
    max_dx, max_dy = max_translate
    dx = random.uniform(-max_dx, max_dx)
    dy = random.uniform(-max_dy, max_dy)
    return image.transform(image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

def shear_image(image, max_shear=0.2):
    shear_factor = random.uniform(-max_shear, max_shear)
    return image.transform(image.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))

def rotate_image(image, max_rotation=30):
    angle = random.uniform(-max_rotation, max_rotation)
    return image.rotate(angle)

def augment_image(image_path, output_path):
    image = Image.open(image_path)
    
    # Apply transformations
    image = translate_image(image)
    image = shear_image(image)
    image = rotate_image(image)
    
    # Save the augmented image
    image.save(output_path)
    print(f"Augmented image saved to {output_path}")

# Example Usage, paths must have .jpeg / .jpg, etc.
input_image_path = ""
output_image_path = ""
augment_image(input_image_path, output_image_path)
