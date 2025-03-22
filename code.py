import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df_expanded = pd.read_csv('dataset.csv')

label_encoder = LabelEncoder()
df_expanded['department_encoded'] = label_encoder.fit_transform(df_expanded['department'])

X_train, X_test, y_train, y_test = train_test_split(
    df_expanded['symptoms'], 
    df_expanded['department_encoded'], 
    test_size=0.3, 
    random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs'],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

labels = list(range(len(label_encoder.classes_)))
target_names = label_encoder.classes_
print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))

conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(label_encoder.classes_)))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

def predict_department(symptoms):
    prediction = best_model.predict([symptoms])
    return label_encoder.inverse_transform(prediction)[0]

example_symptoms = "stomach ache"
predicted_department = predict_department(example_symptoms)
print(f"Predicted Department: {predicted_department}")
