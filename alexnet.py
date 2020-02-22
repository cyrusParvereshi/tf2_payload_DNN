# alexnet.py

# import tflearn
# from tflearn.layers.conv import conv_2d, max_pool_2d
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.estimator import regression
# from tflearn.layers.normalization import local_response_normalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet80x60():
    model = Sequential()
    model.add(Conv2D(96, (5, 5), padding='valid', input_shape=(80,60,3,))) #60,80,3 for RGB
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(256, (5, 5), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Conv2D(384, (3, 3), padding='valid'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(384, (3, 3), padding='valid'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), padding='valid'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(3))#number of outputs (moves)
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

model = alexnet80x60()
model.summary()





#original model

#bare tfLearn by Sentdex, probably won't use because tflearn kind of deprecated
def alexnet(width, height, lr):   #network architecture: 1 input layer, 2 CNN layers with max pool,
# 2 w/out pool, 1 with pool, 3 dense layers, 2 of which are followed by 50% dropout layers. Relu activation 
#used in convolutions, tanh activation for dense layers, regression output layer uses momentum optimizer with 
#loss measured by categorical crossentropy
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')

    model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model

