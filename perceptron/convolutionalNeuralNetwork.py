import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.utils import np_utils

# seed the random number generator
seed = 7
np.random.seed(seed)

(X,y), (Xp, yp) = mnist.load_data()

X = X.reshape(X.shape[0], 1, 28, 28).astype('float32')
Xp = Xp.reshape(Xp.shape[0], 1, 28, 28).astype('float32')

# normalize
X /= 255.0
Xp /= 255.0

y = np_utils.to_categorical(y)
yp = np_utils.to_categorical(yp)

num_classes = yp.shape[1]

def CNN_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='relu'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model
    
model = CNN_model()

model.fit(X,y,validation_data=(Xp,yp), nb_epoch=10, batch_size=200, verbose=2)
scores = model.evaluate(Xp,yp, verbose=0)

print 'CNN Error %.1f%%' % (100-scores[1])*100