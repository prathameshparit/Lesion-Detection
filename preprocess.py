import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import matplotlib
# Importing Necessary Libraries
from skimage import data
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt

matplotlib.use('Agg')

image_path = "test/Late_Blight_106.jpg"


# ----------------------------- Data Augmentation(Filtration) ----------------------------------
def filteration(image_path):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, [224, 224])

    img = img / 255.

    image = tf.expand_dims(img, 0)
    # plt.figure(figsize=(10, 10))


    for i in range(9):
        augmented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0])
        plt.axis("off")
        plt.savefig("static/assets/display/augmented.jpg")


# ----------------------------- Resizing (Preprocess) ----------------------------------

def processing(image_path):
    global original_size, resized_shape, res_img, result
    # loading image
    # Getting 3 images_uploaded to work with
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.cast(img, tf.float32) / 255.
    original = img
    original_size = img[0].shape
    print('Original size', img[0].shape)

    # print(img[0].shape)

    # --------------------------------
    # setting dim of the resize
    res_img = []
    for i in range(len(img)):
        res = tf.io.read_file(image_path)
        res = tf.io.decode_image(res)
        res = tf.image.resize(res, [224, 224])
        res = tf.cast(res, tf.float32) / 255.
        # res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    # Checcking the size
    resized_shape = res_img[1].shape
    print("RESIZED", res_img[1].shape)

    # Visualizing one of the images_uploaded in the array
    result = res_img[1]
    display_pp(tf.cast(original, tf.float32), tf.cast(result, tf.float32))


# Display one image
def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.show()


# Display two images_uploaded
def display_pp(a, b):


    title1 = f"Original Size={original_size}"
    title2 = f"Edited Size={resized_shape}"
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2), plt.grid()
    plt.xticks([]), plt.yticks([])
    plt.savefig("static/assets/display/resize.jpg")


# ----------------------------- RGB to HSV (Preprocess) ----------------------------------





# Setting the plot size to 15,15
def hsv(image_path):
    # plt.figure(figsize=(15, 15))
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, [224, 224])

    image = img / 255.

    # image = tf.expand_dims(img, 0)
    plt.subplot(1, 2, 1)
    # plt.imshow(image)
    hsv_image = tf.image.grayscale_to_rgb(img)
    hsv_image = hsv_image / 255.
    # hsv_image = rgb2hsv(image)
    # plt.subplot(1, 2, 2)
    # hsv_image_colorbar = plt.imshow(hsv_image)
    # plt.colorbar(hsv_image_colorbar, fraction=0.046, pad=0.04)
    display(image, hsv_image)



def display(a, b, title1="Original", title2="Grayscale"):


    plt.subplot(121), plt.imshow(a), plt.title(title1)
    # plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    # plt.xticks([]), plt.yticks([])
    plt.savefig("static/assets/display/hsv.jpg")

# image_path = "static/images_uploaded/second.JPG"
# original = processing(image_path)
# hsv(image_path)
# harris(image_path)
# processing(image_path)
# filteration(image_path)
