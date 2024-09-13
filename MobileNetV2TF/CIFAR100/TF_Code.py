import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_datasets as tfds
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix

filename = f'output{datetime.datetime.now()}'

def save_model(model, filepath: str = "", name: str | None = None):
    '''
    Save the model to a file
    
    :param model: The model to serialize
    :param filepath: The path to save the model to
    :param name: The name of the file to save the model to

    :return: None

    See [reference](https://www.tensorflow.org/guide/keras/serialization_and_saving) for more information
    '''
    if name is None:
        name = ""
        today = str(datetime.datetime.now().date())
        time = str(datetime.datetime.now().time()).split(".")[0].replace(":", "-")
        name = f"model_{today}_{time}.keras"
    model.save(filepath + name)

f = open(filename, "a")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the Food101 dataset
print("Loading dataset")
(ds_train_val, ds_test), ds_info = tfds.load(
    'food101',
    split=['train', 'validation'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

train_size = int(len(ds_train_val)*0.8)

print("Splitting dataset")
shuffle_buffer_size = 10000
ds_train_val = ds_train_val.shuffle(buffer_size=shuffle_buffer_size)
ds_train = ds_train_val.take(train_size)
ds_train = ds_train.shuffle(buffer_size=min(shuffle_buffer_size, train_size))
ds_val = ds_train_val.skip(train_size)
ds_val = ds_val.shuffle(buffer_size=min(shuffle_buffer_size, len(ds_val)))

# Preprocess the dataset
print("Preprocessing dataset")
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_mobilenet(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Apply MobileNetV2-specific preprocessing
    return image, label

def resize_img(image, label):
    return tf.image.resize(image, (224, 224)), label

ds_train = ds_train.map(preprocess_mobilenet)
ds_train = ds_train.map(resize_img)
ds_val = ds_val.map(preprocess_mobilenet)
ds_val = ds_val.map(resize_img)
ds_test = ds_test.map(preprocess_mobilenet)
ds_test = ds_test.map(resize_img)

ds_train = ds_train.batch(16)
ds_val = ds_val.batch(16)
ds_test = ds_test.batch(16)

model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=1.0,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=101,
    classifier_activation='softmax'
)

# Compile the model
model.compile(
              optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # Sparse since labels are not one-hot encoded
              metrics=['accuracy']
              )

f.write(f'Starting Training\n')
f.write(f'Start Time: {datetime.datetime.now()}\n')

# Train the model
history = model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=30
)

f.write(f'Starting Eval\n')
f.write(f'Start Time: {datetime.datetime.now()}\n')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(ds_test)
print(f"Test Accuracy: {test_accuracy * 100:.5f}%\n")
f.write(f"Test Accuracy: {test_accuracy * 100:.5f}%\n")
print(f"Test Loss: {test_loss:.5f}%\n")
f.write(f"Test Loss: {test_loss:.5f}%\n")

print("Saving model\n")
save_model(model)    

f.write(f'Starting Prediction\n')
f.write(f'Start Time: {datetime.datetime.now()}\n')

# Get model predictions
y_true = np.concatenate([y for x, y in ds_test], axis=0)  # True labels from the test set
y_pred = model.predict(ds_test)  # Predicted probabilities
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Extract true positives, false positives, true negatives, and false negatives
TP = np.diag(cm)  # Diagonal values represent true positives for each class
FP = np.sum(cm, axis=0) - TP  # Column sum minus true positives gives false positives
FN = np.sum(cm, axis=1) - TP  # Row sum minus true positives gives false negatives
TN = np.sum(cm) - (FP + FN + TP)  # Total sum minus TP, FP, and FN gives true negatives

print("Confusion Matrix:")
f.write("Confusion Matrix:")
print(cm)
f.write(f'{cm}')
f.write("\n")

print(f"True Positives (TP) SUM: {sum(TP)}")
f.write(f"True Positives (TP) SUM: {sum(TP)}\n")
print(f"True Positives (TP): {TP}")
f.write(f"True Positives (TP): {TP}\n")

print(f"False Positives (FP) SUM: {sum(FP)}")
f.write(f"False Positives (FP) SUM: {sum(FP)}\n")
print(f"False Positives (FP): {FP}")
f.write(f"False Positives (FP): {FP}\n")

print(f"False Negatives (FN) SUM: {sum(FN)}")
f.write(f"False Negatives (FN) SUM: {sum(FN)}\n")
print(f"False Negatives (FN): {FN}")
f.write(f"False Negatives (FN): {FN}\n")

print(f"True Negatives (TN) SUM: {sum(TN)}")
f.write(f"True Negatives (TN) SUM: {sum(TN)}\n")
print(f"True Negatives (TN): {TN}")
f.write(f"True Negatives (TN): {TN}\n")

f.write(f'End Time: {datetime.datetime.now()}\n')

f.close();
