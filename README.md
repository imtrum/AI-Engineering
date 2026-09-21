# AI Engineering Roadmap
This repository contain my AI Engineering learning journey

### GĐ1 — Programming & AI Engineering
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Git
- SQL
- APIs

### GĐ2 — Mathematics for AI
- Linear Algebra
- Calculus
- Probability
- Statistics
- Optimization

## Progress

- [x] Day 1 — NumPy, Pandas, Matplotlib
- [x] Day 2 — Scikit-learn fundamentals
- [x] Day 3 — Preprocessing & Pipelines
- [x] Day 4 — Model Evaluation & Linear Algebra
- [x] Day 5 - Use database, connect to pgAdmin 4
- [x] Day 6 - Async, Await, type hints
- [x] Day 7 - Real data 
    + plot two figure
    + inspect the data
## Dataset exploration
- [x] steps to process data [num,cat]-> pipeline -> column_transformer(preprocessor) -> model_pipeline
- [x] some function, syntax to 

## Environment

This project uses:

- Python
- uv
- Git
- scikit-learn
- pandas
- numpy
- matplotlib

##Data
Numerical features:
- `Age`
- `SibSp`
- `Parch`
- `Fare`

Categorical features:
- `Pclass`
- `Embarked`
- `Sex`

## Machine Learning Pipeline

The project uses the following preprocessing steps:

1. Fill missing numerical values using the median
2. Fill missing categorical values using the most frequent value
3. Encode categorical features using OneHotEncoder
4. Train classification models

## I trained and compared three classification models:
   - Logistic Regression
   - Decision Tree
   - Random Forest 

 + I evaluated the models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   F1 is useful when I want to consider both precision and recall instead of relying on accurcy
   