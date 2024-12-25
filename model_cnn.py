from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Load the data
train_df = pd.read_csv("objects_part2.csv")
test_df = pd.read_csv("objects_part1.csv")

y_train = train_df.pop('label')  # Use pop to avoid deleting separately
y_test = test_df.pop('label')

# Binarize the labels
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.transform(y_test)  # Use transform instead of fit_transform

x_train = train_df.values
x_test = test_df.values

# Reshape the data to include three channels
x_train = x_train.reshape(-1, 56, 56, 3)
x_test = x_test.reshape(-1, 56, 56, 3)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Example of enabling small rotation for augmentation
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(x_train)

# Learning rate reduction
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=0.00001)

# Define the model
model = Sequential()
model.add(Conv2D(35, (7,7), strides=1, padding='same', input_shape=(56, 56, 3)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Conv2D(20, (5,5), strides=1, padding='same'))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Conv2D(10, (3,3), strides=1, padding='same'))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dense(6, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(datagen.flow(x_train, y_train, batch_size=64), epochs=10, validation_data=(x_test, y_test), callbacks=[early_stopping])


# Save the model
model.save('object.h5')
