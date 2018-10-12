import tensorflow as tf
import numpy as np
from math import ceil, floor
import cv2

IMAGE_SIZE=640
def data_augmentation(images):
    aug_data = []
    g = tf.Graph()
    with g.as_default():
        X_resize = tf.placeholder(tf.int32, [None, None, 3])
        tf_img = tf.image.resize_images(X_resize, (640, 640),
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img_flip=tf.image.random_flip_left_right(tf_img)
        img_flip_1=tf.image.random_flip_up_down(img_flip)
        img_bright=tf.image.random_brightness(img_flip_1,
                             max_delta=63)
        img_contrast=tf.image.random_contrast(img_bright,
                                 lower=0.2, upper=1.8)
        img_norm = tf.image.per_image_standardization(img_contrast)
        img_norm.set_shape([640,640,3])
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        for img in images:
            resized_img = sess.run(tf_img, feed_dict={X_resize: img})
            aug_data.append(resized_img)
            # print(resized_img.shape)
    # nt(data[0].shape)
    return np.asarray(aug_data,dtype=np.float32)

def data_augmentation_test(images):
    aug_data = []
    g = tf.Graph()
    with g.as_default():
        X_resize = tf.placeholder(tf.int32, [None, None, 3])
        tf_img = tf.image.resize_images(X_resize, (640, 640),
                                        tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        crop = tf.image.central_crop(tf_img,1)
        #img_flip=tf.image.random_flip_left_right(tf_img)
        #img_bright=tf.image.random_brightness(img_flip,
        #                       max_delta=63)
        #img_contrast=tf.image.random_contrast(img_bright,
        #                         lower=0.2, upper=1.8)
        #img_norm = tf.image.per_image_standardization(tf_img)
        #img_norm.set_shape([480,480,3])
    with tf.Session(graph=g) as sess:
        sess.run(tf.global_variables_initializer())
        for x in images:
            resized_img = sess.run(crop, feed_dict={X_resize: x})
            aug_data.append(resized_img)
            # print(resized_img.shape)
    # nt(data[0].shape)
    return np.asarray(aug_data,dtype=np.float32)


def central_scale_images(X_imgs, Y_imgs,scales):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype=np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale  # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype=np.float32)
    box_ind = np.zeros((len(scales)), dtype=np.int32)
    crop_size = np.array([640, 640], dtype=np.int32)
    graph1=tf.Graph()
    X_scale_data = []
    Y_scale_data=[]
    with graph1.as_default():

    #tf.reset_default_graph()
        X = tf.placeholder(tf.float32, shape=(1, 640, 640, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
        tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session(graph=graph1) as sess:
        sess.run(tf.global_variables_initializer())

        for  img_data,y in zip(X_imgs,Y_imgs):
            batch_img = np.expand_dims(img_data, axis=0)
            scaled_imgs = sess.run(tf_img, feed_dict={X: batch_img})
            X_scale_data.extend(scaled_imgs)
            Y_scale_data.extend([y])
            Y_scale_data.extend([y])
            Y_scale_data.extend([y])


    X_scale_data = np.array(X_scale_data, dtype=np.float32)
    return X_scale_data,np.asarray(Y_scale_data)


def rotate_images(X_imgs,Y_imgs):
    X_rotate = []
    Y_rotate=[]
    graph2=tf.Graph();
    with graph2.as_default():
        X = tf.placeholder(tf.float32, shape=(640, 640, 3))
        k = tf.placeholder(tf.int32)
        tf_img = tf.image.rot90(X, k=k)
    with tf.Session(graph=graph2) as sess:
        sess.run(tf.global_variables_initializer())
        for img,y in zip(X_imgs,Y_imgs):
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict={X: img, k: i + 1})
                X_rotate.append(rotated_img)
                Y_rotate.extend([y])

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate,np.asarray(Y_rotate)


def get_translate_parameters(index):
    if index == 0:  # Translate left 20 percent
        offset = np.array([0.0, 0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = 0
        w_end = int(ceil(0.8 * IMAGE_SIZE))
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 1:  # Translate right 20 percent
        offset = np.array([0.0, -0.2], dtype=np.float32)
        size = np.array([IMAGE_SIZE, ceil(0.8 * IMAGE_SIZE)], dtype=np.int32)
        w_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = IMAGE_SIZE
    elif index == 2:  # Translate top 20 percent
        offset = np.array([0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = 0
        h_end = int(ceil(0.8 * IMAGE_SIZE))
    else:  # Translate bottom 20 percent
        offset = np.array([-0.2, 0.0], dtype=np.float32)
        size = np.array([ceil(0.8 * IMAGE_SIZE), IMAGE_SIZE], dtype=np.int32)
        w_start = 0
        w_end = IMAGE_SIZE
        h_start = int(floor((1 - 0.8) * IMAGE_SIZE))
        h_end = IMAGE_SIZE

    return offset, size, w_start, w_end, h_start, h_end


def translate_images(X_imgs,Y_imgs):
    offsets = np.zeros((len(X_imgs), 2), dtype=np.float32)
    n_translations = 4
    X_translated_arr = []
    Y_translated_arr= []
    graph_translate=tf.Graph()
    #tf.reset_default_graph()
    with tf.Session(graph=graph_translate) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_translations):
            X_translated = np.zeros((len(X_imgs), IMAGE_SIZE, IMAGE_SIZE, 3),
                                    dtype=np.float32)
            X_translated.fill(1.0)  # Filling background color
            base_offset, size, w_start, w_end, h_start, h_end = get_translate_parameters(i)
            offsets[:, :] = base_offset
            glimpses = tf.image.extract_glimpse(X_imgs, size, offsets)

            glimpses = sess.run(glimpses)
            X_translated[:, h_start: h_start + size[0], \
            w_start: w_start + size[1], :] = glimpses
            X_translated_arr.extend(X_translated)
    for i in range(4):
        for y in Y_imgs:
            Y_translated_arr.extend([y])


    X_translated_arr = np.array(X_translated_arr, dtype=np.float32)
    return X_translated_arr,np.asarray(Y_translated_arr)


def rotate_images_at_fined_angle(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.contrib.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


def flip_images(X_imgs,Y_imgs):
    X_flip = []
    Y_flip=[]
    flip_graph=tf.Graph()
    #tf.reset_default_graph()
    with flip_graph.as_default():
        X = tf.placeholder(tf.float32, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
        tf_img1 = tf.image.flip_left_right(X)
        tf_img2 = tf.image.flip_up_down(X)
        tf_img3 = tf.image.transpose_image(X)
    with tf.Session(graph=flip_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for img,y in zip(X_imgs,Y_imgs):
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
            Y_flip.extend([y]*3)
    X_flip = np.array(X_flip, dtype = np.float32)
    return X_flip,np.asarray(Y_flip)


def add_salt_pepper_noise(X_imgs):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = X_imgs.copy()
    row, col, _ = X_imgs_copy[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_imgs_copy:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_imgs_copy


def add_gaussian_noise(X_imgs,Y_imgs):
    gaussian_noise_imgs = []
    gaussian_y=[]
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img,Y_img in zip(X_imgs,Y_imgs):
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
        gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0)
        gaussian_noise_imgs.append(gaussian_img)
        gaussian_y.extend([Y_img])
    gaussian_noise_imgs = np.array(gaussian_noise_imgs, dtype=np.float32)
    return gaussian_noise_imgs,np.asarray(gaussian_y)


def get_mask_coord(imshape):
    vertices = np.array([[(0.09 * imshape[1], 0.99 * imshape[0]),
                          (0.43 * imshape[1], 0.32 * imshape[0]),
                          (0.56 * imshape[1], 0.32 * imshape[0]),
                          (0.85 * imshape[1], 0.99 * imshape[0])]], dtype = np.int32)
    return vertices


def get_perspective_matrices(X_img):
    offset = 15
    img_size = (X_img.shape[1], X_img.shape[0])

    # Estimate the coordinates of object of interest inside the image.
    src = np.float32(get_mask_coord(X_img.shape))
    dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                      [img_size[0] - offset, img_size[1]]])

    perspective_matrix = cv2.getPerspectiveTransform(src, dst)
    return perspective_matrix
def perspective_transform(X_img):
    # Doing only for one type of example
    perspective_matrix = get_perspective_matrices(X_img)
    warped_img = cv2.warpPerspective(X_img, perspective_matrix,
                                     (X_img.shape[1], X_img.shape[0]),
                                     flags = cv2.INTER_LINEAR)
    return warped_img