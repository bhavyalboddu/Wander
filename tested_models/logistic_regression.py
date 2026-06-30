from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def train_model(train_prepared, train_labels):
    model = LogisticRegression(max_iter=10000, random_state=0)
    model.fit(train_prepared, train_labels)
    return model

def evaluate_model(model, test_prepared, test_labels):
    classifier_predictions = model.predict(test_prepared)
    accuracy = accuracy_score(test_labels, classifier_predictions)*100
    return accuracy, classifier_predictions

def generate_classification_report(classifier_predictions, test_labels):
    return classification_report(test_labels, classifier_predictions)

def tune_model(train_prepared, train_labels, test_prepared, test_labels, cv=5):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=100000, solver='saga', tol=1e-4)
    )
    param_grid = {
        'logisticregression__C': [0.1, 1, 10, 100],
        'logisticregression__penalty': ['l1', 'l2']
    }
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=cv)
    grid_search.fit(train_prepared, train_labels)
 
    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    test_predictions = best_model.predict(test_prepared)
    test_acc = accuracy_score(test_labels, test_predictions) * 100
 
    return grid_search, best_model, test_acc

def full_pipeline(train_prepared, train_labels, test_prepared, test_labels):
    model = train_model(train_prepared, train_labels)
    accuracy, classifier_predictions = evaluate_model(model, test_prepared, test_labels)
    report = generate_classification_report(classifier_predictions, test_labels)
    return model, accuracy, report