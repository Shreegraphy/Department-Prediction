# Department Prediction

## Overview

This project uses machine learning to predict the appropriate medical department based on patient symptoms. The model is trained on a dataset containing various symptoms and their corresponding departments.

## Features

- Uses **TF-IDF Vectorization** to process text-based symptoms.
- Implements **Logistic Regression** for classification.
- Utilizes **GridSearchCV** for hyperparameter tuning.
- Provides a function to predict the medical department based on user-input symptoms.
- Outputs a **classification report** and **confusion matrix** for model evaluation.

## Installation

Ensure you have Python and the required dependencies installed. You can install them using:

```sh
pip install pandas scikit-learn matplotlib
```

## Usage

1. Place your dataset in the same directory as `dataset.csv`.
2. Run the script:

   ```sh
   python department_prediction.py
   ```

3. To predict a department based on symptoms:

   ```python
   predict_department("stomach ache, nausea")
   ```

## Dataset Format

The dataset should be in a CSV file named `dataset.csv` with the following columns:

- `symptoms`: Text description of symptoms.
- `department`: The corresponding medical department.

### Example:

```csv
symptoms,department
"chest pain, shortness of breath",Cardiology
"abdominal pain, bloating",Gastroenterology
```

## Model Performance

The model is evaluated using:

- **Classification Report**: Precision, recall, and F1-score for each department.
- **Confusion Matrix**: Visualization of true vs. predicted labels.

## Contributing

Feel free to submit issues or pull requests to improve the model and dataset.
