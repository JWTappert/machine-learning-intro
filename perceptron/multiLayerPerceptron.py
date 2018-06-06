import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# seed the random number generator
seed = 7
np.random.seed(seed)

(X,y), (Xp, yp) = mnist.load_data()

num_pixels = X.shape[1] * X.shape[2]

X = X.reshape(X.shape[0], num_pixels).astype('float32')
Xp = Xp.reshape(Xp.shape[0], num_pixels).astype('float32')

# normalize
X /= 255.0
Xp /= 255.0

y = np_utils.to_categorical(y)
yp = np_utils.to_categorical(yp)

num_classes = yp.shape[1]

def MLP_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = MLP_model()

model.fit(X,y,validation_data=(Xp,yp), nb_epoch=10, batch_size=200, verbose=2)
scores = model.evaluate(Xp,yp, verbose=0)

print 'MLP Error %.2f%%' % (100-scores[1])*100