import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from keras import layers
from keras.models import load_model


# Funcion que optimiza los datasets
def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
                num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)

def plotResultTraining():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


# Se carga el dataset
dataset_name = 'tf_flowers'
(train_ds, val_ds, test_ds), metadata = tfds.load(
    dataset_name,
    split=['train[:80%]', 'train[80:90%]', 'train[90%:]'],
    # split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True,
)

# Parametros del dataset y de entrenamiento
AUTOTUNE = tf.data.AUTOTUNE
img_width, img_height = 150, 150  # Tamaño de imagen para el entrenamiento
batch_size = 32
epochs = 10
# learning_rate = 0.0005
num_classes = metadata.features['label'].num_classes
get_label_name = metadata.features['label'].int2str
classes = []

# Normaliza y cambia el tamaño de las imagenes
resize_and_rescale = tf.keras.Sequential([
    layers.Rescaling(1. / 255),
    layers.Resizing(img_width, img_height)
])

# Hace aumento de datos sobre el modelo
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

for i in range(num_classes):
    classes.append(get_label_name(i))

print(f'El numero de clases es: {num_classes}')
print(classes)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)

#Arquitectura de la red neuronal
modeloCNN_DA = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', ),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

modeloCNN_DA.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# Entrenamiento del modelo
history = modeloCNN_DA.fit(train_ds, validation_data=val_ds, epochs=epochs)
plotResultTraining()

loss, acc = modeloCNN_DA.evaluate(test_ds)
print("Accuracy", acc)

modeloCNN_DA.save(f'pruebas/RedCNN_{dataset_name}.hdf5')