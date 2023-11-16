# General
from tqdm import tqdm
from os.path import exists
from os import listdir

# Data Pre-Proccessing
import numpy as np
from cv2 import imread, resize
from random import randint, sample, choice
from sklearn.model_selection import train_test_split

# For Model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv3D, Dense, Flatten, AveragePooling3D, Dropout

if not exists("last_chunk.dat"):
    with open("last_chunk.dat", "w") as file:
        file.write("0")

class Phrex:
    def __init__(self) -> None:
        CHUNK_SIZE = 10

        folders = listdir("dataset")

        self.CHUNKS = [folders[i:i + CHUNK_SIZE]
                       for i in range(0, len(folders), CHUNK_SIZE)]

        self._set_memory_growth()

        self.model = self._build_model()

        self._train()

    def _set_memory_growth(self):
        # Set memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def _build_model(self) -> Sequential:
        model = Sequential()

        model.add(Input(shape=(11, 100, 100, 1)))

        model.add(Conv3D(32, kernel_size=(2, 2, 2), activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2)))
        model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu'))
        model.add(AveragePooling3D(pool_size=(2, 2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='RMSprop', metrics=['accuracy'])

        model.summary()

        return model

    def _fetch_dataset(self, chunk) -> list:
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

        print(f"{memory_usage} MB Used By Dataset")

        return dataset

    def _getRandom(self, x, l) -> int:
        index = x

        while x == index:
            index = randint(0, l-1)

        return index

    def _proccess_data(self, dataset) -> list:
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
                    feature = choice(
                        dataset[self._getRandom(index, len(dataset))])

                    labels.append(0)

                features[-1].append(feature)

        features = np.array(features)
        features = features.reshape(
            (-1, features.shape[1], features.shape[2], features.shape[3], 1))

        memory_usage = round(
            features.nbytes / (1024 * 1024),
            2
        )

        print(f"{memory_usage} MB Used By Features")

        labels = np.array(labels)

        return train_test_split(features, labels, test_size=0.2, random_state=42)

    def _train(self) -> None:
        print("Fetching Last Chunk Index...")

        with open("last_chunk.dat", "r") as file:
            chunk_index = int(file.read())

        if chunk_index:
            print("Loading Previous Weights...")
            self.model.load_weights("weights.h5")

            print(f"Weights Loaded From Chunk #{chunk_index}")

        for chunk in self.CHUNKS[chunk_index:]:
            dataset = self._fetch_dataset(chunk)

            x_train, x_test, y_train, y_test = self._proccess_data(dataset)

            self.model.fit(x_train, y_train, validation_data=(
                x_test, y_test), epochs=2, batch_size=16)

            self.model.save_weights("weights.h5")

            with open("last_chunk.dat", "w") as file:
                file.write(str(self.CHUNKS.index(chunk)+1))

        self.model.save("phrex_model")

Phrex()