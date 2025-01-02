from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import time

# Load the data
train_df = pd.read_csv("objects_part2.csv")
test_df = pd.read_csv("objects_part1.csv")

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
    zoom_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=True
)

datagen.fit(x_train)

# Learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.000001)

# Define a simpler model
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(56, 56, 3)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2))

model.add(Conv2D(32, (5, 5), padding='same'))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=5, validation_data=(x_test, y_test), callbacks=[early_stopping, learning_rate_reduction])

# Save the model
model_name = "object_" + time.strftime("%Y%m%d-%H%M%S") + ".h5"
model.save(model_name)
