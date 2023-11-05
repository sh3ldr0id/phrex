import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from os import listdir
from cv2 import imread, resize
from tqdm import tqdm
from random import randint, sample, choice
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set memory growth 
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load Dataset
dataset = []

folders = listdir("dataset")[:2]

for folder in tqdm(folders):
    files = listdir(f"dataset/{folder}")

    images = []

    for file in files:
        image = imread(f"dataset/{folder}/{file}", 0)

        image = resize(image, (100, 100))

        images.append(image)

    dataset.append(images)

memory_usage = round(
    np.array(
        dataset, 
        dtype=np.uint8
    ).nbytes / (1024 * 1024), 
    2
)
print(f"{memory_usage} MB Used")

# Process Dataset
def getRandom(x):
    index = x

    while x == index:
        index = randint(0, len(dataset)-1)

    return index

features = []
labels = []

for index in range(len(dataset)):
    data = dataset[index]

    for i in range(64):
        shuffled = sample(data, len(data))

        features.append(shuffled[:63])

        if i % 2 == 0:
            feature = choice(shuffled)

            labels.append(1)

        else:
            feature = choice(dataset[getRandom(index)])

            labels.append(0)

        features[-1].append(feature)

features = np.array(features)
features = features.reshape((-1, features.shape[1], features.shape[2], features.shape[3], 1))

labels = np.array(labels)

print(f"{len(labels)} Labels")

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build Model
model = keras.Sequential()

model.add(layers.Input(shape=(features.shape[1], features.shape[2], features.shape[3], features.shape[4])))

model.add(layers.Conv3D(32, kernel_size=(5, 5, 5), activation='relu'))
model.add(layers.MaxPooling3D(pool_size=(5, 5, 5)))
model.add(layers.Conv3D(64, kernel_size=(5, 5, 5), activation='relu'))
model.add(layers.MaxPooling3D(pool_size=(5, 5, 5)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=1)
# model.fit(features, labels, epochs=10, batch_size=128)

# # Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy}")
