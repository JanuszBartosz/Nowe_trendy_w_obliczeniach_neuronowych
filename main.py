from keras import layers, metrics
from keras import models, optimizers
from keras.applications import VGG16
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

from utils import *

train_dir = '/home/snowflake/Desktop/train'
validation_dir = '/home/snowflake/Desktop/validate'

nTrain = 300
nVal = 60
nClasses = 3
batch_size = 10

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

print("**********Getting train features************")
train_features, train_labels, train_generator = get_features_and_labels(nTrain,
                                                                        nClasses,
                                                                        batch_size,
                                                                        train_dir,
                                                                        conv_base)

print("**********Getting validation features************")
validation_features, validation_labels, validation_generator = get_features_and_labels(nVal,
                                                                                       nClasses,
                                                                                       batch_size,
                                                                                       validation_dir,
                                                                                       conv_base)

print("**********Configure model for training************")
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nClasses, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=[metrics.categorical_accuracy])

print("**********Start of training************")
history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features, validation_labels)
                    )

print("**********End of training************\n\n")

fnames = validation_generator.filenames

ground_truth = [np.where(r == 1)[0][0] for r in validation_labels]

label2index = validation_generator.class_indices

# Getting mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)

# errors = nVal - accuracy_score(ground_truth, predictions, normalize=False)
errors = np.where(predictions != ground_truth)[0]
accuracy = accuracy_score(ground_truth, predictions)
precision, recall, f_score, support = score(ground_truth, predictions, average='weighted')

print("No of errors = {}/{}".format(len(errors), nVal))
print("Accuracy: {}".format(accuracy))
print("Weighted average Precision: {}".format(precision))
print("Weighted average Recall: {}".format(recall))
print("Weighted average F-score: {}".format(f_score))

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
