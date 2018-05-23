import os
from math import ceil

from sklearn.model_selection import train_test_split
import numpy as np

all_features = np.load(open("/media/snowflake/Data/features.dat", "rb"))
all_labels = np.load(open("/media/snowflake/Data/labels.dat", "rb"))

train_dir = '/home/snowflake/Documents/nowe_trendy/train'
path, dirs, files = os.walk(train_dir).__next__()

nTrain = sum([len(files) for r, d, files in os.walk(train_dir)]) - len(dirs)
val_size = 0.176
nVal = ceil(nTrain * val_size)

seed = 7
np.random.seed(seed)
train_features, validation_features, train_labels, validation_labels = train_test_split(all_features,
                                                                                        all_labels,
                                                                                        test_size=val_size,
                                                                                        random_state=seed)

f = open("/media/snowflake/Data/train_features.dat", "wb+")
np.save(f, train_features)
f = open("/media/snowflake/Data/train_labels.dat", "wb+")
np.save(f, train_labels)

f = open("/media/snowflake/Data/validation_features.dat", "wb+")
np.save(f, validation_features)
f = open("/media/snowflake/Data/validation_labels.dat", "wb+")
np.save(f, validation_labels)
