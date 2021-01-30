from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

dataset_fashion = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = dataset_fashion.load_data()

print("Train images:", len(train_images))
print("Test images:", len(test_images))

print(train_images[1].shape)
print(train_images[1])

plt.imshow(train_images[1], cmap=plt.cm.binary);
plt.show()

print(train_labels[1])

unique_labels = np.unique(train_labels)
print("total unique labels", len(unique_labels))


label_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(label_names[train_labels[1]])

train_images = train_images / 255
test_images = test_images / 255

print(np.min(train_images))
print(np.max(train_images))

def show_prediction(number):
    prediction = saved_model.predict(test_images[number].reshape(1, 28, 28))
    prediction_name = label_names[np.argmax(prediction)]
    actual_name = label_names[test_labels[number]]
    plt.imshow(test_images[number], cmap=plt.cm.binary);
    plt.title("Actual: {} & Predicted: {}".format(actual_name, prediction_name))
    plt.show()

try:
    print("Loading model ...")
    saved_model = keras.models.load_model("model.h5")
    predictions = saved_model.predict(test_images)
    print(predictions[10])
    print(np.argmax(predictions[10]))
    print(label_names[np.argmax(predictions[10])])
    plt.imshow(test_images[10], cmap=plt.cm.binary);
    plt.show()

    saved_model.evaluate(x=test_images, y=test_labels)

    for number in [5668, 214, 789]:
        show_prediction(number)

except IOError as e:
    print("Creating model ...")
    model = keras.Sequential()

    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    print(model.summary())

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x=train_images, y=train_labels, epochs=5)
    model.save("model.h5")
