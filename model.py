import csv
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Constants
steering_offset = 0.3
data_path = "data/"

# Load the data and offset the steering on left and right images.
def get_image_path_and_labels(data_file):
    img_paths, steering_angles = [], []
    with open(data_file) as fin:
        skip_next_entry = True
        for center_img, left_img, right_img, steering_angle, throttle, break_power, speed in csv.reader(fin):
            # The first entry is just the header so skip it.
            if skip_next_entry:
                skip_next_entry = False
                continue
            # Add the center, left, and right images paths.
            img_paths += [center_img.strip(), left_img.strip(), right_img.strip()]
            # Append steering offset and add the angle.
            steering_angles += [float(steering_angle), float(steering_angle) + steering_offset, float(steering_angle) - steering_offset]
    return img_paths, steering_angles

# Process the image
def process_image(image_path, steering_angle):
    # Compress the size to 100x100 so we can train faster.
    image = load_img(image_path, target_size=(100,100,3))
    image = img_to_array(image)
    return image, steering_angle

# Generator
def generator(batch_size, x, y):
    while 1:
        batch_x, batch_y = [], []
        for i in range(batch_size):

            index = random.randint(0, len(x) - 1)
            steering_angle = y[index]
            image, steering_angle = process_image(data_path + x[index], steering_angle)
            batch_x.append(image)
            batch_y.append(steering_angle)

            # Also add to the batch a flipped version of the image.
            image_flipped = np.fliplr(image)
            steering_angle_flipped = -steering_angle
            batch_x.append(image_flipped)
            batch_y.append(steering_angle_flipped)

        yield np.array(batch_x), np.array(batch_y)

# Define the training model.
def model(shape):

    # We must use SAME padding so the output size isn't reduced too small before flattening the network.
    border_mode = 'same'
    model = Sequential()

    # Normalize the input.
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=shape, output_shape=shape))

    model.add(Convolution2D(24, 5, 5, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D())

    model.add(Convolution2D(36, 5, 5, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D())

    model.add(Convolution2D(48, 5, 5, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D())

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D())

    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode=border_mode))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.35))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.35))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.35))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer="adam")

    return model

# Train the model.
def train():
    net = model(shape=(100,100,3))

    # Print the strucutre of the network
    for layer in net.layers:
        print(layer, layer.output_shape)

    # Get the image paths, and steering angles.
    x, y = get_image_path_and_labels(data_path + 'driving_log.csv')

    # Shuffle the data.
    x, y = shuffle(x, y, random_state=42)

    # Split into training and validation sets.
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)

    # Train the model.
    net.fit_generator(generator(64, x_train, y_train),
                      validation_data=generator(64, x_val, y_val),
                      nb_val_samples=12000,
                      samples_per_epoch=48000,
                      nb_epoch=3)

    # Save the model.
    net.save('model.h5')

# Activate this script
if __name__ == '__main__':
    train()
