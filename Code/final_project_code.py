# Gaberial Campese
# Fernando Zambrano
# ML2 Final Project

# Import Libraries

import numpy as np
import pandas as pd
import keras
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import cv2

weights_path = '/home/ubuntu/fp/data/resnet50_weights.h5'
top_model_weights_path = '/home/ubuntu/fp/datafc_model.h5'

# Hyperparameters
batch_size = 8
EPOCHS = 30

# Read in Labels from CSV
df_train = pd.read_csv('/home/ubuntu/fp/data/labels/trainLabels.csv')

print(df_train.head())

targets_series = pd.Series(df_train['level'])
one_hot = pd.get_dummies(targets_series, sparse = True)

one_hot_labels = np.asarray(one_hot)
# image sizing set to 256
img_size1 = 256
img_size2 = 256

x_train = []
y_train = []
x_test = []

i = 0
for j, level in tqdm(df_train.values):
    if type(cv2.imread('/home/ubuntu/fp/data/train/train_subset/{}.jpeg'.format(j))) == type(None):
        continue
    else:
        img = cv2.imread('/home/ubuntu/fp/data/train/train_subset/{}.jpeg'.format(j))
        label = one_hot_labels[i]
        x_train.append(cv2.resize(img, (img_size1, img_size2)))
        y_train.append(label)
        i += 1
np.save('/home/ubuntu/fp/data/x_train', x_train)
np.save('/home/ubuntu/fp/data/y_train', y_train)
print('Done')

x_train = np.load('/home/ubuntu/fp/data/x_train.npy')
y_train = np.load('/home/ubuntu/fp/data/y_train.npy')


y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=1)


# initialize data augmentor
augmentor = ImageDataGenerator()


augmentor = ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest")


classes = y_train.shape[1]

base_model = ResNet50(weights = "imagenet", include_top=False, input_shape=(img_size1, img_size2, 3))
print("base_model = resnet50")

# Add a new top layer classifier
x = base_model.output
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dense(16, activation='relu')(x)
predictions = Dense(classes, activation='sigmoid')(x)#softmax

# Training Model
model = Model(inputs=base_model.input, outputs=predictions)
x.load_weights(top_model_weights_path)
model.summary()
print("This is the number of trainable weights before freezing the resenet50 weights:", len(model.trainable_weights))
# Freezing the first 45 resnet50 layers so the pretrained weights do not get updated, effectivley rendering the model useless

#for layer in base_model.layers: #turn this on
    #layer.trainable = True
print("This is the number of trainable weights after freezing the resnet50 weights:", len(model.trainable_weights))


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=1e-4), metrics=['accuracy'])#optimizer = sgd # rmsprop

H = model.fit_generator(
    augmentor.flow(X_train, Y_train,batch_size = batch_size),
    validation_data=(X_valid, Y_valid),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=EPOCHS,callbacks=[ModelCheckpoint("resnet_e10_sgd.hdf5",
                                             monitor="val_loss",
                                             save_best_only=True)])
# evaluate the network
print("Evaluating network...")
predictions = model.predict(X_valid, batch_size = batch_size)

print("Cohen Kappa: ", cohen_kappa_score(np.argmax(model.predict(X_valid), axis=1),np.argmax(Y_valid, axis = 1)))
print("F1 score: ", f1_score(np.argmax(model.predict(X_valid), axis=1), np.argmax(Y_valid, axis=1), average = 'macro'))

# Plot training/testing loss

N = np.arange(0, EPOCHS)

history_dict = H.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
print(history_dict.keys())
accuracy = history_dict['accuracy']

EPOCHS = range(1, len(accuracy) + 1)

plt.plot(EPOCHS, loss_values, 'bo', label='Training Loss')
plt.plot(EPOCHS, val_loss_values, 'b', label='Validation Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('graph5.png')
plt.show()