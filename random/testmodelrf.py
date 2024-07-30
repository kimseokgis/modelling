def evaluate_model(model, X_test, y_test, label_encoder):

def main():
    
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