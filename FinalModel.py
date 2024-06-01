import os
import numpy as np
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.applications import VGG19, ResNet152V2
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, concatenate, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import SGD, Adam
from keras._tf_keras.keras.preprocessing.image import img_to_array, load_img



# Paths and parameters
data_dir = "/Users/aaravbansal/Developer/Research Bootcamp/"
train_dir = '/Users/aaravbansal/Developer/Research Bootcamp/Traindata'
val_dir = '/Users/aaravbansal/Developer/Research Bootcamp/valdata'
img_size = (224, 224)
batch_size = 32
num_classes = 8

# Create training and validation directories
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

def preprocess_image(image):
    image = cv2.resize(image, img_size)
    image = cv2.medianBlur(image, 9)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

def preprocess_segmented_image(image):
    clahe = cv2.createCLAHE(clipLimit=3.8, tileGridSize=(6, 6))
    image = cv2.resize(image, img_size)
    image = cv2.medianBlur(image, 9)
    # Example segmentation using simple thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray  = clahe.apply(gray)
    _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    output_image = np.zeros_like(image)
    output_image[mask == 0] = image[mask == 0]
    return output_image

def image_generator(directory, batch_size, img_size, num_classes):
    datagen = ImageDataGenerator()
    generator = datagen.flow_from_directory(directory,
                                            target_size=img_size,
                                            batch_size=batch_size,
                                            class_mode='categorical')
    while True:
        batch_x, batch_y = next(generator)
        batch_x1 = np.zeros_like(batch_x)
        batch_x2 = np.zeros_like(batch_x)
        for i in range(batch_x.shape[0]):
            img = batch_x[i].astype(np.uint8)  # Ensure the image is in the correct format
            batch_x1[i] = preprocess_image(img)
            batch_x2[i] = preprocess_segmented_image(img)
        yield ((batch_x1, batch_x2), batch_y)
def create_dataset(generator, batch_size, img_size, num_classes, a):
    output_signature = (
        (tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32),
         tf.TensorSpec(shape=(None, *img_size, 3), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
    )
    return tf.data.Dataset.from_generator(lambda: generator(a, batch_size, img_size, num_classes),
                                          output_signature=output_signature)

train_dataset = create_dataset(image_generator, batch_size, img_size, num_classes, train_dir)
val_dataset = create_dataset(image_generator, batch_size, img_size, num_classes, val_dir)
# Model creation
input_shape = (224,224,3)

input1 = Input(shape=input_shape)
base_model1 = VGG19(include_top=False, weights='imagenet', input_tensor=input1)
for layer in base_model1.layers:
    layer.trainable = False
x1 = base_model1.output
x1 = Conv2D(128, (3, 3), activation='relu', padding='same')(x1)
x1 = MaxPooling2D((2, 2))(x1)
x1 = Flatten()(x1)

input2 = Input(shape=(224,224,3))
base_model2 = ResNet152V2(include_top=False, weights='imagenet', input_tensor=input2)
for layer in base_model2.layers:
    layer.trainable = False
x2 = base_model2.output
x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Flatten()(x2)

# Concatenate paths
x = concatenate([x2, x1])
x = Dense(1024, activation='softmax')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=[input1, input2], outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
train_gen = image_generator(train_dir, batch_size, img_size, num_classes)
val_gen = image_generator(val_dir, batch_size, img_size, num_classes)

model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=25)

# Fine-tuning
for layer in base_model1.layers:
    layer.trainable = True
for layer in base_model2.layers:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset,
          validation_data=val_dataset,
          epochs=25)
