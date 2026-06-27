from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_model(train_prepared, train_labels):
    model = SVC(kernel="poly", gamma=10, C=1)
    model.fit(train_prepared, train_labels)
    return model

def evaluate_model(model, test_prepared, test_labels):
    classifier_predictions = model.predict(test_prepared)
    accuracy = accuracy_score(test_labels, classifier_predictions)*100
    return accuracy, classifier_predictions

def generate_classification_report(classifier_predictions, test_labels):
    return classification_report(test_labels, classifier_predictions)