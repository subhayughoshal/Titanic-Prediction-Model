# Titanic Survival Prediction

## *Overview*
This project predicts passenger survival on the Titanic using machine learning models, implemented in a Jupyter notebook (`titanic_subhayu.ipynb`). The dataset is from the *Kaggle Titanic competition*, and the goal is to predict whether a passenger survived based on features like class, sex, age, and fare. The project includes *data cleaning*, *exploratory data analysis (EDA)*, *feature engineering*, *model training*, and *evaluation*, culminating in predictions for the test set.

## *Dataset*
The dataset consists of:
- *train.csv*: Training data (891 passengers) with features (e.g., `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`) and the target (`Survived`: 0 = died, 1 = survived).
- *test.csv*: Test data (418 passengers) with the same features but no `Survived` column.
- *gender_submission.csv*: A sample submission assuming all females survive and males die.
- *best_model_predictions.csv*: Output predictions from the best-performing model.

## *Project Steps*
1. **Data Cleaning**:
   - Filled missing `Age` values with the median (28.0, from train set).
   - Filled missing `Embarked` values with the mode ('S').
   - Filled missing `Fare` in test set with train set median (14.4542).
   - Dropped `Cabin` due to excessive missing values.
   - Handled safely to avoid errors on repeated runs.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized survival rates by:
     - **Sex**: Females had a 74.2% survival rate, males 18.9%.
     - **Pclass**: 1st class (63.0%) > 2nd class (47.3%) > 3rd class (24.2%).
     - **Age**: Survivors slightly younger (mean 28.3 vs. 30.0 for non-survivors).
     - **Fare**: Survivors paid higher fares (mean $48.40 vs. $22.12).
     - **Embarked**: Cherbourg (55.4%) > Queenstown (39.0%) > Southampton (33.9%).
   - Generated a correlation heatmap showing strong relationships (e.g., `Sex` and `Survived`: 0.54).

3. **Feature Engineering**:
   - Added `FamilySize` (`SibSp` + `Parch` + 1).
   - Encoded `Sex` (male=0, female=1) and `Embarked` (S=0, C=1, Q=2).
   - Normalized numerical features (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`) using `StandardScaler`.

4. **Model Training**:
   - Trained four models: *Logistic Regression*, *Random Forest*, *Extra Trees*, and *XGBoost*.
   - Evaluated using:
     - **Validation accuracy** on a 20% holdout set.
     - **5-fold cross-validation (CV)** mean score on the full training set.
   - Selected the best model based on the average of validation accuracy and CV mean.

5. **Predictions and Submission**:
   - Generated predictions on `test.csv` using the best model.
   - Saved predictions to `best_model_predictions.csv` for Kaggle submission.
   - Compared predictions to `gender_submission.csv` (if available) for agreement (~70-90%).

## *Dependencies*
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`
- Install dependencies:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost
