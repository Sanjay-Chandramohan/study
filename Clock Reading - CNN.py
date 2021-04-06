#import csv
import keras
import numpy as np
import random
#import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Activation, MaxPooling2D, Dropout, Input, BatchNormalization

data = np.load('C:/Users/Sanjay/Desktop/IDL/Assignment2/labels.npy')
data1 = data[0:, 0:1]
data2 = data[0:, 1:]

channel = 1
image_size = 150
path = np.load('C:/Users/Sanjay/Desktop/IDL/Assignment2/images.npy')

def predict(ind):        
        
    im = Image.fromarray(path[ind])
    #print(im)
    plt.imshow(im)
    #print('Input image:')
    plt.show()
    im = im.convert('L')
    im = im.resize((image_size,image_size), Image.ANTIALIAS)
    im = np.array(im)
    im = processing_img(im).reshape((1, image_size, image_size, channel))
    #print(im)
    correct = str(data1[ind])+':'+str(data2[ind])
    time = model.predict(im)
    
    hour = np.argmax(time[0])
    minute = int(time[1][0][0])
    print('Detected Time:', str(hour)+':'+str(minute))
    print('Correct Time:', correct )

def processing_img(img):
    
    img = img/255
    img -= .5
    return img

def image_queue(ids, batch_size):

    image_batch = np.zeros((batch_size, image_size, image_size, 1))
    hour_label = np.zeros((batch_size, 1))
    minute_label = np.zeros((batch_size, 1))
    batch_ids = np.random.choice(ids, batch_size)
    ind = 0
    
    for ind in range(len(batch_ids)):
        
        index = batch_ids[ind]
        img = Image.fromarray(path[batch_ids[ind]])
        img = img.resize((image_size,image_size), Image.ANTIALIAS)
        img = np.array(img)
        image_batch[ind] = processing_img(img).reshape((image_size, image_size, 1))
        hour_label[ind] = (data1[index])
        minute_label[ind] = (data2[index])
        ind += 1
            
    return (np.array(image_batch), np.array(hour_label), np.array(minute_label))

inp = Input(shape=(image_size,image_size, 1))

a = Conv2D(25, kernel_size=3, strides=2, activation=None, padding='same')(inp)
a = MaxPooling2D(pool_size=(2, 2), strides=2)(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)

a = Conv2D(50, kernel_size=5, strides=2, activation=None, padding='same')(a)
a = MaxPooling2D(pool_size=(2, 2))(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)

a = Conv2D(100, kernel_size=3, strides=1, activation=None, padding='same')(a)
a = MaxPooling2D(pool_size=(2, 2))(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)

a = Conv2D(200, kernel_size=3, strides=1, activation=None, padding='same')(a)
a = MaxPooling2D(pool_size=(2, 2))(a)
a = BatchNormalization()(a)
a = Activation('relu')(a)
a = Dropout(0.3)(a)

a = Flatten()(a)

minute = Dense(100, activation='relu')(a)
minute = Dense(200, activation='relu')(minute)
minute = Dense(1, activation='linear', name='minute')(minute)

hour = Dense(144, activation='relu')(a)
hour = Dense(144, activation='relu')(hour)
hour = Dense(12, activation='softmax', name='hour')(hour)



model = Model(inputs=inp, outputs=[hour, minute])


x = list(range(0,18000))

train_ids = np.array(random.sample(x,14400))
z = train_ids.tolist()

for element in z:
    if element in x:
        x.remove(element)
test_ids = np.array(x)

np.random.shuffle(train_ids)
np.random.shuffle(test_ids)


x_train, y1_train, y2_train = image_queue(train_ids, 14400)
x_test, y1_test, y2_test = image_queue(test_ids, 3600)

opti = keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(loss=['sparse_categorical_crossentropy', 'mae'], optimizer=opti, metrics=['accuracy', 'mae'])

model.fit(x_train, [y1_train, y2_train], epochs=10, batch_size=200, validation_data=(x_test, [y1_test, y2_test]))

predict(713)


