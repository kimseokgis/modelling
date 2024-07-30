def main():
    
    
    
    

    

    

    

    

    

    

    # Train RandomForest model
    rf_classifier = train_random_forest(X_train_tfidf, y_train, n_estimators, random_state)

    # Evaluate the model
    evaluate_model(rf_classifier, X_test_tfidf, y_test, label_encoder)

    # Save the model, vectorizer, and label encoder
    save_model(rf_classifier, tfidf_vectorizer, label_encoder, path)

if __name__ == "__main__":
    main()