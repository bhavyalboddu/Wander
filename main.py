import sys
import data_processing

if len(sys.argv) < 2:
    print("Error: Missing required argument.")
    sys.exit(1)

model_name = sys.argv[1]

train_prepared, test_prepared, train_labels, test_labels, full_pipeline = data_processing.prepare_data()

if model_name == "cnn":
    from tested_models import cnn
    model, test_loss, test_acc = cnn.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
elif model_name == "neural_network_tensorflow":
    from tested_models import neural_network_tensorflow
    model, test_loss, test_acc = neural_network_tensorflow.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
elif model_name == "neural_network_scikit_learn":
    from tested_models import neural_network_scikit_learn
    model, test_acc, report = neural_network_scikit_learn.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
    print("Classification Report: ")
    print(report)
elif model_name == "naive_bayes":
    from tested_models import naive_bayes
    model, test_acc, report = naive_bayes.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
    print("Classification Report: ")
    print(report)
elif model_name == "logistic_regression":
    from tested_models import logistic_regression
    model, test_acc, report = logistic_regression.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
    print("Classification Report: ")
    print(report)
elif model_name == "svm":
    from tested_models import svm
    model, test_acc, report = svm.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
    print("Classification Report: ")
    print(report)
elif model_name == "decision_tree":
    from tested_models import decision_tree
    model, test_acc, report = decision_tree.full_pipeline(train_prepared, train_labels, test_prepared, test_labels)
    print("Test Accuracy: " + str(test_acc))
    print("Classification Report: ")
    print(report)