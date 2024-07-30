def evaluate_model(model, X_test, y_test, label_encoder):
    
    
    
    
    
    
    
    
        
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