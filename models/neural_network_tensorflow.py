import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

def scale_model(train_prepared, test_prepared):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_prepared)
    X_test = scaler.transform(test_prepared)
    return X_train, X_test

def create_model(X_train):
    model = Sequential([
        Dense(64, activation="relu", kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    return model

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(model, X_train, train_labels, X_test, test_labels):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    model.fit(
        X_train, train_labels,
        validation_data=(X_test, test_labels),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    return model

def evaluate_model(model, X_test, test_labels):
    test_loss, test_acc = model.evaluate(X_test, test_labels) * 100
    return test_loss, test_acc

def full_pipeline(train_prepared, train_labels, test_prepared, test_labels):
    X_train, X_test = scale_model(train_prepared, test_prepared)
    model = create_model(X_train, train_labels, X_test, test_labels)
    model = compile_model(model)
    model = train_model(model, X_train, train_labels, X_test, test_labels)
    test_loss, test_acc = evaluate_model(model, X_test, test_labels)
    return model, test_loss, test_acc