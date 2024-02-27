import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

from sklearn.model_selection import train_test_split

def cifar10_model_pipeline_func(X,y,num_classes = 10, epochs = 10, batch_size = 32, lr = 0.001, optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']):
    """
    This function takes in a dataframe and returns a model pipeline.
    """
    # Split the dataframe into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Create the model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                        input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    # Fit the model
    model.fit(x_train,y_train,epochs = epochs, batch_size = batch_size, validation_data = (x_val,y_val))

    # Return the model
    return model


def plot_model_accuracy(history):
    """
    This function plots the accuracy of the model.
    """
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig('model_accuracy.png')
    