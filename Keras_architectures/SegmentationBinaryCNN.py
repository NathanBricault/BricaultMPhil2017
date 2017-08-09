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
from PreProcess import load_training_data
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.constraints import maxnorm
import os
import numpy as np
import re

from PreProcess import Dataset
from PreProcess import DownSampleParams

def default_callbacks(output_prefix, dperiod = 1):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.1,
                                  patience=5, min_lr=0.001)
    checkpoint = ModelCheckpoint(output_prefix + "_lweights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=dperiod)
    checkpoint_best = ModelCheckpoint(output_prefix + "_BEST_lweights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=dperiod)
    csv_logger = CSVLogger(output_prefix + '_log.csv', append=True, separator=';')
    return( [reduce_lr, checkpoint, checkpoint_best, csv_logger] )

def kumar_cnn(model_function, dataset_name, model_name,
                     num_epochs = 115,
                     batch_size = 128,
                     image_dims = (51, 51, 3),
                     downsample_params = DownSampleParams(),
                     callback_function = default_callbacks,
                     metric_list = ['accuracy']):
    target_dir_prefix = os.environ["OUTPUT_DIR"] + model_name + "/"
    output_prefix = target_dir_prefix + model_name
    if os.path.exists(target_dir_prefix):
        print("WARNING: target dataset already exists, this process is overwriting it!")               
    else:
        os.makedirs(target_dir_prefix)   

    callback_list = callback_function(output_prefix)

    # the data, shuffled and split between train and test sets
    dataset = Dataset(dataset_name)
    dataset.load()
    if downsample_params.do_downsample:
        dataset = dataset.downsample(downsample_params.num_training, downsample_params.num_validation)
    data = dataset.unpack()
        
    (x_train, y_train), (x_test, y_test), input_shape = prep_data(data, image_dims)
 
    model = Sequential()
    model_function(model, input_shape)
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr = 0.01),
                  metrics=metric_list)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              verbose=1,
              validation_data=(x_test, y_test), 
              callbacks = callback_list)
    score = model.evaluate(x_test, y_test, verbose=0)
    
    write_metrics(model, score, metric_list, print_output = True, output_prefix = output_prefix)
    model.save(output_prefix +  "_final.h5")
    

def kumar_binary_architecture(model, input_shape):
    kumar_architecture(model, input_shape, output_channels = 2)

def kumar_ternary_architecture(model, input_shape):
    kumar_architecture(model, input_shape, output_channels = 3)

def kumar_architecture(model, input_shape, output_channels):
    model.add(Conv2D(25, kernel_size=(4,4),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(50, (5,5), activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(80, (6,6), activation='relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(output_channels,   activation='softmax'))


def kumar_constrained_architecture(model, input_shape, output_channels = 3):
    model.add(Conv2D(25, kernel_size=(4,4),
                     activation='relu',
                     input_shape=input_shape, kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(50, (5,5), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(80, (6,6), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    
    model.add(Dense(output_channels,   activation='softmax', kernel_constraint=maxnorm(3)))


def prep_data(data, image_dims):

    (img_rows, img_cols, channels) = image_dims

    (x_train, y_train), (x_test, y_test) = data
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
 
    num_classes = len(np.unique(y_train))
    # convert class vectors to binary class matrices
    if num_classes == 2:   # NOTE: the /2 is hack, because we have 0 and 2 as values
        y_train = y_train/2
        y_test = y_test/2
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return( ((x_train, y_train), (x_test, y_test), input_shape ))

def print_model(model):
    for layer in model.layers:
        print("\t**************************")
        print("\t" + str(layer.input_shape))
        print("\t" + str(layer.output_shape))

def print_data_check(y_train, y_test):
    print(sum(y_train)/len(y_train))
    print(sum(y_test)/len(y_test))
    print(y_train.shape)

## TODO: this is just a stub
def predictions( model, data_x ):
    predictions = model.predict(data_x, batch_size = 128)
    return(predictions)


# TODO: implement this validation
def validate_class_distribution():
    return True

def write_metrics(model, score, metrics, print_output = False, output_prefix = "~/"):
    if print_output:
        print(", ".join(model.metrics_names))
        print(model.metrics_names)
    target = open(output_prefix + "final_metrics.txt", 'w')

    target.write("POTATO")
    target.write("\n")
    target.close()

def load_saved_model( model_name, mode ):
    if mode == 'best':
        print("Loading best model for " + model_name)
        files = [f for f in os.listdir(os.environ["OUTPUT_DIR"] + model_name) if re.match(r'.*_BEST_lweights.[0-9]+-.*.hdf5',f)]
        p = re.compile('.*_BEST_lweights\.([0-9]+)-.*\.hdf5')
        indices = [p.match(x).groups()[0] for x in files]
        ind_array = np.array(indices)
        max_index = ind_array.argmax(axis = 0)
        best_model = files[max_index]
        print(best_model)
        model = keras.models.load_model(os.environ["OUTPUT_DIR"] + model_name + "/" + best_model)
    elif mode == 'final':
        print("Loading final model for " + model_name)
        model = keras.models.load_model(os.environ["OUTPUT_DIR"] + model_name + "/" + model_name + "_final.h5")
    else: 
        print("Bad mode provided: " + str(mode) + " for model: " + model_name)

    return(model) 
