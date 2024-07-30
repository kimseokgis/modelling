def evaluate_model(model, X_test, y_test, label_encoder):

def main():

    # Load saved model, vectorizer, and label encoder
    
    
    
    
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