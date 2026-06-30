import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt

def prepare_data_for_cnn(train_prepared, test_prepared):
    train_prepared_reshaped = np.expand_dims(train_prepared, axis=-1)
    test_prepared_reshaped = np.expand_dims(test_prepared, axis=-1)
    return train_prepared_reshaped, test_prepared_reshaped

def create_model(hp, input_shape):
    model = Sequential()
    # 1st layer, using 1d cnn because less data
    filters_1 = hp.Int('filters_1', min_value=16, max_value=64, step=16)
    kernel_size_1 = hp.Choice('kernel_size_1', values=[2, 3, 5])
    model.add(Conv1D(filters=filters_1,
                     kernel_size=kernel_size_1,
                     padding="same",
                     activation="relu",
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # 2nd layer
    filters_2 = hp.Int('filters_2', min_value=32, max_value=128, step=32)
    kernel_size_2 = hp.Choice('kernel_size_2', values=[2, 3, 5])
    model.add(Conv1D(filters=filters_2,
                     kernel_size=kernel_size_2,
                     padding="same",
                     activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    # 3rd layer
    filters_3 = hp.Int('filters_3', min_value=32, max_value=128, step=32)
    kernel_size_3 = hp.Choice('kernel_size_3', values=[2, 3, 5])
    model.add(Conv1D(filters=filters_3,
                     kernel_size=kernel_size_3,
                     padding="same",
                     activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    # dense layer with tunable units - prepping for hyperparam tuning
    dense_units = hp.Int('dense_units', min_value=128, max_value=1024, step=128)
    model.add(Dense(dense_units, activation='relu'))


    dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def build_callbacks(es_patience=5, lr_patience=3):
    early_stop = EarlyStopping(monitor="val_loss", patience=es_patience, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=lr_patience, verbose=1)
    return [early_stop, lr_scheduler]

def build_tuner(input_shape, directory='hyper_tuning_dir', project_name='cnn_tuning_two', max_trials=20):
    tuner = kt.RandomSearch(
        lambda hp: create_model(hp, input_shape),
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=1,
        directory=directory,
        project_name=project_name
    )
    return tuner

def run_tuner_search(tuner, train_reshaped, train_labels, test_reshaped, test_labels,
                     epochs=50, batch_size=32, callbacks=None):
    if callbacks is None:
        callbacks = build_callbacks()
 
    tuner.search(
        train_reshaped, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_reshaped, test_labels),
        callbacks=callbacks
    )
    return tuner

def evaluate_best_model(tuner, test_reshaped, test_labels):
    model_best_performance = tuner.get_best_models(num_models=1)[0]
    test_loss, test_acc = model_best_performance.evaluate(test_reshaped, test_labels)
    return model_best_performance, test_loss, test_acc

def full_pipeline(train_prepared, train_labels, test_prepared, test_labels,
                  epochs=50, batch_size=32, tuner_max_trials=20):
    train_reshaped, test_reshaped = prepare_data_for_cnn(train_prepared, test_prepared)
    input_shape = (train_reshaped.shape[1], train_reshaped.shape[2])
    tuner = build_tuner(input_shape, max_trials=tuner_max_trials)
    tuner = run_tuner_search(tuner, train_reshaped, train_labels, test_reshaped, test_labels,
                             epochs=epochs, batch_size=batch_size)
    best_model, test_loss, test_acc = evaluate_best_model(tuner, test_reshaped, test_labels)
    return best_model, test_loss, test_acc