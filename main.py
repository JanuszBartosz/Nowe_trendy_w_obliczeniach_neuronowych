import os
import time
from math import floor, ceil

import numpy
from keras import layers, metrics
from keras import models, optimizers
from keras.applications import VGG16
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils import *
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

start = time.time()

train_dir = '/home/snowflake/Documents/nowe_trendy/train'
test_dir = '/home/snowflake/Documents/nowe_trendy/test'

path, dirs, files = os.walk(train_dir).__next__()

val_size = 0.15
nTrain = sum([len(files) for r, d, files in os.walk(train_dir)]) - len(dirs)
nVal = ceil(nTrain * val_size)
path, dirs, files = os.walk(test_dir).__next__()
nTest = sum([len(files) for r, d, files in os.walk(test_dir)]) - len(dirs)
nClasses = len(dirs)
batch_size = 20
dense_size = 256
epochs = 10

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

print("**********Getting train features************")
all_features, all_labels, all_generator = get_features_and_labels(nTrain,
                                                                  nClasses,
                                                                  batch_size,
                                                                  train_dir,
                                                                  conv_base,
                                                                  False)

seed = 7
numpy.random.seed(seed)
train_features, validation_features, train_labels, validation_labels = train_test_split(all_features,
                                                                                        all_labels,
                                                                                        test_size=val_size,
                                                                                        random_state=seed)

print("**********Getting test features************")
test_features, test_labels, test_generator = get_features_and_labels(nTest,
                                                                     nClasses,
                                                                     batch_size,
                                                                     test_dir,
                                                                     conv_base)

print("**********Configure model for training************")
model = models.Sequential()
model.add(layers.Dense(dense_size, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nClasses, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=[metrics.categorical_accuracy])

print("**********Start of training************")
history = model.fit(train_features,
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels))

print("**********End of training************\n\n")

ground_truth = [np.where(r == 1)[0][0] for r in test_labels]

predictions = model.predict_classes(test_features)
prob = model.predict(test_features)

# errors = nVal - accuracy_score(ground_truth, predictions, normalize=False)
errors = np.where(predictions != ground_truth)[0]
accuracy = accuracy_score(ground_truth, predictions)
precision, recall, f_score, support = score(ground_truth, predictions, average='weighted')

print("No of errors = {}/{}".format(len(errors), nTest))
print("Accuracy: {}".format(accuracy))
print("Weighted average Precision: {}".format(precision))
print("Weighted average Recall: {}".format(recall))
print("Weighted average F-score: {}".format(f_score))

end = time.time()
print("Time: " + str(end - start))

# fnames = validation_generator.filenames
# label2index = validation_generator.class_indices
#
# # Getting mapping from class index to class label
# idx2label = dict((v, k) for k, v in label2index.items())
#
# for i in range(len(errors)):
#     pred_class = np.argmax(prob[errors[i]])
#     pred_label = idx2label[pred_class]
#
#     print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#         fnames[errors[i]].split('/')[0],
#         pred_label,
#         prob[errors[i]][pred_class]))
#
#     original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
#     plt.imshow(original)
#     plt.show()
