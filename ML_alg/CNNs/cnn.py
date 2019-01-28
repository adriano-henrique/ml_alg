# Part 1: Building the CNN

# Importing the packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend as K
import numpy as np
K.set_image_dim_ordering('th')
# Part 1: Initializing the CNN
classifier = Sequential()

# Step 1: Convolution
classifier.add(Conv2D(32, kernel_size = (3,3),activation = 'relu', input_shape = (3, 64, 64)))

# Step 2: Pooling Step
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a secondo Convolution Layer
classifier.add(Conv2D(32, kernel_size = (3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2: Fitting your images to the CNN
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')

     
test_set = test_datagen.flow_from_directory('dataset/test_set',
        target_size = (64, 64),
        batch_size = 32,
        class_mode = 'binary')
        

classifier.fit_generator(training_set,
        steps_per_epoch = (8000/32),
        epochs = 25,
        validation_data = test_set,
        validation_steps = (2000/32))

# Making the prediction
test_image1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))
test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1,axis=0)
result = classifier.predict(test_image1)
training_set.class_indices

test_image2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64,64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)
result2 = classifier.predict(test_image2)
training_set.class_indices



        