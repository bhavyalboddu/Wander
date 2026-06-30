from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_model(train_prepared, train_labels):
    model = MultinomialNB()
    model.fit(train_prepared, train_labels)
    return model

def evaluate_model(model, test_prepared, test_labels):
    classifier_predictions = model.predict(test_prepared)
    accuracy = accuracy_score(test_labels, classifier_predictions)*100
    return accuracy, classifier_predictions

def generate_classification_report(classifier_predictions, test_labels):
    return classification_report(test_labels, classifier_predictions)

def full_pipeline(train_prepared, train_labels, test_prepared, test_labels):
    model = train_model(train_prepared, train_labels)
    accuracy, classifier_predictions = evaluate_model(model, test_prepared, test_labels)
    report = generate_classification_report(classifier_predictions, test_labels)
    return model, accuracy, report