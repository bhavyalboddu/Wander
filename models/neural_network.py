from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def train_model(train_prepared, train_labels):
    model = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam", max_iter=500, random_state=41)
    model.fit(train_prepared, train_labels)
    return model

def evaluate_model(model, test_prepared, test_labels):
    classifier_predictions = model.predict(test_prepared)
    accuracy = accuracy_score(test_labels, classifier_predictions)*100
    return accuracy, classifier_predictions

def generate_classification_report(classifier_predictions, test_labels):
    return classification_report(test_labels, classifier_predictions)

def tune_model(model, train_prepared, train_labels, test_prepared, test_labels, cv=5):

    param_grid = {
        "hidden_layer_sizes": [(32,), (32, 16), (64, 32), (128, 64, 32)],  # Different layer structures
        "activation": ["relu", "tanh"],  # Activation functions
        "solver": ["adam"],  # Only using Adam optimizer
        "alpha": [0.0001, 0.001, 0.01],  # L2 Regularization
        "learning_rate": ["constant", "adaptive"],  # Learning rate strategies
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=2)
    grid_search.fit(train_prepared, train_labels)
 
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(test_prepared)
    test_acc = accuracy_score(test_labels, test_predictions) * 100
 
    return grid_search, best_model, test_acc