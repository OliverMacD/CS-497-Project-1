import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# Load the MNIST dataset
(ds_train, ds_test), ds_info = tfds.load(
    'malaria',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def reshape_img(image, label):
  return tf.reshape(image, (224, 224, 3)), label

def resize_img(image, label):
  return tf.image.resize(image, (224, 224)), label

ds_train = ds_train.map(
    resize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.map(
  reshape_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(8) ############################################## Batch sizes
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    resize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(
    reshape_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(8)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=1.0,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=2,
    classifier_activation='softmax'
)

# Compile the model
model.compile(
              optimizer=Adam(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',  # Sparse since labels are not one-hot encoded
              metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=6
    #callbacks=[early_stopping]
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(ds_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")