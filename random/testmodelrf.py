def prepare_test_data(test_dataset, tfidf_vectorizer, label_encoder):
    """
    Prepare test data by combining questions and answers, vectorizing text, and encoding labels.
    """
    questions = test_dataset.iloc[:, 0].values.tolist()
    answers = test_dataset.iloc[:, 1].values.tolist()
    combined_text = [f"{q} {a}" for q, a in zip(questions, answers)]
    
    X_test_tfidf = tfidf_vectorizer.transform(combined_text)
    y_test = label_encoder.transform(answers)
    
    return X_test_tfidf, y_test

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model's performance on the test set and log the results.
    """
    logging.info("Evaluating model on test data")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logging.info(f"Accuracy: {accuracy}")
    logging.info("Classification Report:\n" + report)
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    
    return accuracy, report, conf_matrix

def main():
    # Set parameters
    test_file_path = 'test_data.csv'
    model_path = 'output_dir_14k/rf_classifier.pkl'
    vectorizer_path = 'output_dir_14k/tfidf_vectorizer.pkl'
    encoder_path = 'output_dir_14k/label_encoder.pkl'
    delimiter = "|"
    header = None
    lineterminator = '\n'

    # Load saved model, vectorizer, and label encoder
    logging.info("Loading saved model, vectorizer, and label encoder")
    rf_classifier = load_pickle(model_path)
    tfidf_vectorizer = load_pickle(vectorizer_path)
    label_encoder = load_pickle(encoder_path)
    
    # Load test dataset
    test_dataset = load_test_dataset(test_file_path, delimiter, header, lineterminator)
    
    # Prepare test data
    X_test_tfidf, y_test = prepare_test_data(test_dataset, tfidf_vectorizer, label_encoder)
    
    # Evaluate the model
    accuracy, report, conf_matrix = evaluate_model(rf_classifier, X_test_tfidf, y_test, label_encoder)
    
    # Save evaluation results
    evaluation_path = "output_dir_14k/evaluation_results.txt"
    with open(evaluation_path, 'w') as eval_file:
        eval_file.write(f"Accuracy: {accuracy}\n")
        eval_file.write("Classification Report:\n")
        eval_file.write(report + "\n")
        eval_file.write(f"Confusion Matrix:\n{conf_matrix}\n")
    
    logging.info(f"Evaluation results saved to: {evaluation_path}")

if __name__ == "__main__":
    main()