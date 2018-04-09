from random import shuffle

import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def get_features_and_labels(n_samples, n_classes, batch_size, data_dir, conv_base):
    data_generator = ImageDataGenerator(rescale=1. / 255,
                                        # horizontal_flip=True,
                                        # fill_mode="nearest",
                                        # zoom_range=0.3,
                                        # width_shift_range=0.3,
                                        # height_shift_range=0.3
                                        )

    features = np.zeros(shape=(n_samples, 7, 7, 512))
    labels = np.zeros(shape=(n_samples, n_classes))

    generator = data_generator.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=shuffle)

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= n_samples:
            break

    features = np.reshape(features, (n_samples, 7 * 7 * 512))

    return features, labels, generator
