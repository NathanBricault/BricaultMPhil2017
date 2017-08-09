'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PreProcess import Dataset
from PreProcess import DownSampleParams
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
import os

def default_callbacks(output_prefix, dperiod = 1):
   # reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=.1,
                              #    patience=5, min_lr=0.001)
    checkpoint = ModelCheckpoint(output_prefix + "_lweights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=dperiod)
    checkpoint_best = ModelCheckpoint(output_prefix + "_BEST_lweights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=dperiod)
    csv_logger = CSVLogger(output_prefix + '_log.csv', append=True, separator=';')
    return( [checkpoint, checkpoint_best, csv_logger] )

def prefix_handler(model_name):
    target_dir_prefix = os.environ["OUTPUT_DIR"] + "MNIST/" +  model_name + "/"
    output_prefix = target_dir_prefix + model_name
    if os.path.exists(target_dir_prefix):
        print("WARNING: target dataset already exists, this process is overwriting it!")               
    else:
        os.makedirs(target_dir_prefix)
    return(output_prefix)

def mnist_cnn( model_name, downsample_params ): 
    batch_size = 128
    num_classes = 10
    epochs = 20
    output_prefix = prefix_handler(model_name)
    # input image dimensions
    img_rows, img_cols = 28, 28
    
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dataset = Dataset("", xt = x_train, yt = y_train, xv = x_test, yv = y_test)
    
    if downsample_params.do_downsample:
        dataset = dataset.downsample(downsample_params.num_training, downsample_params.num_validation)
    ((x_train, y_train), (x_test, y_test)), input_shape = prep_data( dataset.unpack(), img_cols, img_rows, num_classes ) 

    model = Sequential()
    architecture(model, input_shape, num_classes)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks = default_callbacks(output_prefix), 
              validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def architecture(model, input_shape, num_classes):
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
def prep_data( data, img_cols, img_rows, num_classes):
    ((x_train, y_train), (x_test, y_test)) = data
    
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    print(len(y_train))
    print(len(y_test))
    #exit()
    return( (((x_train, y_train), (x_test, y_test)), input_shape) )
