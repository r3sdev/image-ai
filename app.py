# Code taken from https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/
import numpy
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import tensorflowjs as tfjs
from keras.datasets import cifar10
import matplotlib.pyplot as plt

MODEL_NAME = "image-classification"
# Set random seed for purposes of reproducibility
RANDOM_SEED = 21
EPOCHS = 5
BATCH_SIZE = 256
OPTIMIZER = 'adam'

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
    
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]

def build_nn():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))

    model.add(Dropout(0.2))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

    print(model.summary())
    return model

try:
    model = load_model(MODEL_NAME)
    print(model)
    loss1, acc1 = model.evaluate(X_test, y_test)
    print(f"Loss is {loss1},\nAccuracy is {acc1*100}")
except IOError as e:
    print("No existing model found")
    model = build_nn()


numpy.random.seed(RANDOM_SEED)
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
model.save("image-classification")

# Model evaluation
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print(model.summary())

plt.style.use('ggplot')
plt.plot(hist.history['loss'], label = 'loss')
plt.plot(hist.history['val_loss'], label='val loss')
plt.title("Loss vs Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# plt.show()
plt.savefig('loss.png')

plt.plot(hist.history['accuracy'], label = 'accuracy')
plt.plot(hist.history['val_accuracy'], label='val accuracy')
plt.title("accuracy vs Val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
# plt.show()
plt.savefig('accuracy.png')