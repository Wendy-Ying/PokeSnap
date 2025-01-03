from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LeakyReLU, ReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import time

# Load the data
train_df = pd.read_csv("objects_single_part2.csv")
test_df = pd.read_csv("objects_single_part1.csv")

y_train = train_df.pop('label')
y_test = test_df.pop('label')

# Binarize the labels
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)

x_train = train_df.values
x_test = test_df.values

# Reshape the data to include three channels
x_train = x_train.reshape(-1, 56, 56, 3)
x_test = x_test.reshape(-1, 56, 56, 3)

# Data augmentation (reduce intensity for small datasets)
datagen = ImageDataGenerator(
    brightness_range=(0.8, 1.2)
)

datagen.fit(x_train)

# Learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=0, verbose=1, factor=0.9, min_lr=1e-9)

# Define a simpler model
model = Sequential()
model.add(Conv2D(16, (7, 7), padding='same', input_shape=(56, 56, 3)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2))

model.add(Conv2D(32, (5, 5), padding='same'))
model.add(ReLU())
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2))

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(ReLU())
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))

# Compile the model
initial_learning_rate = 2e-5
optimizer = Adam(learning_rate=initial_learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=5, validation_data=(x_test, y_test), callbacks=[early_stopping, learning_rate_reduction])

# Save the model
model_name = "object_" + time.strftime("%Y%m%d_%H%M%S") + ".h5"
model.save(model_name)
