#train_model.py
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from alexnet import alexnet80x60
#from keras.applications.mobilenet import MobileNet

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 2
MODEL_NAME = f'tf2payload-heavy-{LR}-alexnetv2-{EPOCHS}-epochs.model'

train_data = np.load('training_data_balanced.npy', allow_pickle = True)
train = train_data[:-200]   
test = train_data[-200:]
train_x = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,3)
train_y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,3)
test_y = [i[1] for i in test]

model = alexnet80x60() #MobileNet(input_shape=(WIDTH,HEIGHT,1), include_top=False)

#opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.5, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#have to do this to avoid list error
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)
model.fit(train_x, train_y, batch_size = 40, epochs = EPOCHS, verbose = 1)

val_loss, val_acc = model.evaluate(test_x, test_y)
print("Val loss: ", val_loss,"Val acc: ", val_acc)

model.save(MODEL_NAME)


# #Alexnet implementation from https://www.mydatahack.com/building-alexnet-with-keras/
# model = Sequential()
# #4728, 80, 60, 1 is the shape of the input data

# # 1st Convolutional Layer
# model.add(Conv2D(filters=96, input_shape=(80, 60, 1), kernel_size=(11,11), strides=(4,4), padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# # Pooling 
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# # Batch Normalisation before passing it to the next layer
# model.add(BatchNormalization())

# # 2nd Convolutional Layer
# model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid', data_format='channels_first'))
# model.add(Activation('relu'))
# # Pooling
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 3rd Convolutional Layer
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', data_format='channels_first'))
# model.add(Activation('relu'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 4th Convolutional Layer
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid', data_format='channels_first'))
# model.add(Activation('relu'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 5th Convolutional Layer
# model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid', data_format='channels_first'))
# model.add(Activation('relu'))
# # Pooling
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# # Batch Normalisation
# model.add(BatchNormalization())

# # Passing it to a dense layer
# model.add(Flatten())
# # 1st Dense Layer
# model.add(Dense(4096, input_shape=(224*224*3,)))
# model.add(Activation('relu'))
# # Add Dropout to prevent overfitting
# model.add(Dropout(0.4))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 2nd Dense Layer
# model.add(Dense(4096))
# model.add(Activation('relu'))
# # Add Dropout
# model.add(Dropout(0.7))
# # Batch Normalisation
# model.add(BatchNormalization())

# # 3rd Dense Layer
# model.add(Dense(1000))
# model.add(Activation('relu'))
# # Add Dropout
# model.add(Dropout(0.7))
# # Batch Normalisation
# model.add(BatchNormalization())

# # Output Layer
# model.add(Dense(17))
# model.add(Activation('softmax'))