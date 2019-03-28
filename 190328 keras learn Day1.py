import tensorflow as tf
from tensorflow import keras
print(keras.__version__)
fashion_mnist=keras.datasets.fashion_mnist
(x1,y1),(x2,y2)=fashion_mnist.load_data()
x1=x1/255
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(128,activation="relu"),keras.layers.Dense(10,activation="softmax")])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["acc"])
model.fit(x1,y1,epochs=10)
