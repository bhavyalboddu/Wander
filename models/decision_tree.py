from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_model(train_prepared, train_labels):
    model = DecisionTreeClassifier(random_state=41)
    model.fit(train_prepared, train_labels)
    return model

def evaluate_model(model, test_prepared, test_labels):
    classifier_predictions = model.predict(test_prepared)
    accuracy = accuracy_score(test_labels, classifier_predictions)*100
    return accuracy, classifier_predictions

def generate_classification_report(classifier_predictions, test_labels):
    return classification_report(test_labels, classifier_predictions)

def tune_model(train_prepared, train_labels, test_prepared, test_labels, cv=5):
    param_grid = {
        'max_depth': range(1, 10, 1),
        'min_samples_leaf': range(1, 20, 2),
        'min_samples_split': range(2, 20, 2),
        'criterion': ["entropy", "gini"]
    }
    tree = DecisionTreeClassifier(random_state=41)
    grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=cv, verbose=True)
    grid_search.fit(train_prepared, train_labels)
 
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(test_prepared)
    test_acc = accuracy_score(test_labels, test_predictions) * 100
 
    return grid_search, best_model, test_acc