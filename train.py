import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import json
import numpy as np
from constant_variables import JSON_PATH, MODEL_PATH, LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_KEYWORDS


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract x and y
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])

    return X, y

def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    # load dataset
    X, y = load_dataset(data_path)

    # -- split --
    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y,
                                                                 test_size = test_size, random_state = 42)
    # validation test split
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,
                                                                 test_size = test_validation, random_state = 42)

    # covert inputs to 2d (#segments, 13) to 3d arrays:
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test


def build_model(input_shape, LEARNING_RATE, error="sparse_categorical_crossentropy"):

    # build nn
    model = keras.Sequential()

    #conv layer 1
    model.add(keras.layers.Conv2D(64, (3,3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001))) #avoid overfitting
    model.add(keras.layers.BatchNormalization())    #speed up training
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding="same")) #downsampling by factor of 2

    #conv layer 2
    model.add(keras.layers.Conv2D(32, (3,3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())    #speed up training
    model.add(keras.layers.MaxPooling2D((3,3), strides=(2,2), padding="same"))

    #conv layer 3
    model.add(keras.layers.Conv2D(32, (2,2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())    #speed up training
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding="same"))

    #flat output feed into Dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3)) #avoid overfitting
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    print(model.summary())

    return model

def main():
    # load train/test/validation split
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = get_data_splits(JSON_PATH)

    # build CNN model
    # 3d input for CNN, (#segments, #coeff, 1)
    #last dim is depth/img channel = 1
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, Y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, Y_test)
    print(f"Test error: {test_error}, test accuracy {test_accuracy}")
    # save the model
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()