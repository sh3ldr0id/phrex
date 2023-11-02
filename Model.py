import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from os import listdir
from cv2 import imread, resize
from tqdm import tqdm
from random import randint, sample, choice

# Load Dataset
dataset = []

folders = listdir("dataset")[:10]

for folder in tqdm(folders):
    files = listdir(f"dataset/{folder}")

    images = []

    for file in files:
        image = imread(f"dataset/{folder}/{file}", 0)

        image = resize(image, (100, 100))

        images.append(image)

    dataset.append(images)
arr = np.array(dataset, dtype=np.uint8)

memory_usage = round(arr.nbytes / (1024 * 1024), 2)
print(f"{memory_usage} MB Used")

# Process Dataset
def getRandom(x):
    index = x

    while x == index:
        index = randint(0, len(dataset))

    return index

features = []
labels = []

for index in range(len(dataset)):
    data = dataset[index]

    for i in range(4):
        shuffled = sample(data, len(data))

        features.append(shuffled[:-63])

        if i % 2 == 0:
            feature = choice(shuffled)

            labels.append(1)

        else:
            feature = choice(dataset[getRandom(index)])

            labels.append(0)

        features[-1].append(feature)
        
# Build Model
model = keras.Sequential()

model.add(layers.Input(shape=(28, 28, 1)))

model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(6, 6), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(4, 4)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
