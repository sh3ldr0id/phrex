import numpy as np
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, AveragePooling2D, Dropout
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

CHUNK_SIZE = 10

# Build Model
model = Sequential()

model.add(Input(shape=(11, 100, 100)))

model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(AveragePooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

folders = listdir("dataset")

folder_chunks = [folders[i:i + CHUNK_SIZE] for i in range(0, len(folders), CHUNK_SIZE)]

for i, chunk in enumerate(folder_chunks):
    print(f"Chunk: {i + 1}")

    dataset = []

    for folder in tqdm(chunk):
        files = listdir(f"dataset/{folder}")

        images = []

        for file in files:
            image = imread(f"dataset/{folder}/{file}", 0)

            image = resize(image, (100, 100)) / 255

            images.append(image)

        dataset.append(images)

    memory_usage = round(
        np.array(
            dataset, 
            dtype=np.uint8
        ).nbytes / (1024 * 1024), 
        2
    )

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

            features.append(shuffled[:10])

            if i % 2 == 0:
                feature = choice(shuffled)

                labels.append(1)

            else:
                feature = choice(dataset[getRandom(index)])

                labels.append(0)

            features[-1].append(feature)

    features = np.array(features)
    print(features.shape)

    # features = features.reshape((-1, features.shape[1], features.shape[2], features.shape[3], 1))

    memory_usage = round(
        features.nbytes / (1024 * 1024), 
        2
    )
    print(f"{memory_usage} MB Used By Features")

    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=128)
    # model.fit(features, labels, epochs=10, batch_size=128)

    model.save_weights("weights.h5")

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(test_accuracy*100, 2)}%")