    Vectorize the text data using TF-IDF.
    """
    logging.info(f"Vectorizing text data with max features {max_features}")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(text_train)
    X_test_tfidf = tfidf_vectorizer.transform(text_test)
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

def train_random_forest(X_train, y_train, n_estimators, random_state):
    """
    Train a RandomForestClassifier.
    """
    logging.info(f"Training RandomForestClassifier with {n_estimators} estimators")
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model and print the accuracy and classification report.
    """
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    logging.info(f"Accuracy: {accuracy}")
    logging.info("Classification Report:\n" + report)
    return accuracy, report

def save_model(model, vectorizer, label_encoder, path):
    """
    Save the model, vectorizer, and label encoder to disk.
    """
    logging.info("Saving model, vectorizer, and label encoder")
    with open(os.path.join(path, 'rf_classifier.pkl'), 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(path, 'tfidf_vectorizer.pkl'), 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with open(os.path.join(path, 'label_encoder.pkl'), 'wb') as encoder_file:
        pickle.dump(label_encoder, encoder_file)

def main():
    # Set parameters
    path = "output_dir_14k/"
    file_path = 'data.csv'
    delimiter = "|"
    header = None
    lineterminator = '\n'
    test_size = 0.2
    random_state = 42
    max_features = 1000
    n_estimators = 100

    # Create output directory
    create_output_directory(path)

    # Load dataset
    dataset = load_dataset(file_path, delimiter, header, lineterminator)

    # Separate questions and answers
    questions = dataset.iloc[:, 0].values.tolist()
    answers = dataset.iloc[:, 1].values.tolist()

    # Combine questions and answers
    combined_text = combine_questions_answers(questions, answers)

    # Encode labels
    labels, label_encoder = encode_labels(answers)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(combined_text, labels, test_size, random_state)

    # Vectorize the text data
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = vectorize_text(X_train, X_test, max_features)

    # Train RandomForest model
    rf_classifier = train_random_forest(X_train_tfidf, y_train, n_estimators, random_state)

    # Evaluate the model
    evaluate_model(rf_classifier, X_test_tfidf, y_test, label_encoder)

    # Save the model, vectorizer, and label encoder
    save_model(rf_classifier, tfidf_vectorizer, label_encoder, path)

if __name__ == "__main__":
    main()