# Programming Assignment 8: Ensemble Learning

## Objective
To implement and compare Random Forest Regression using a from-scratch implementation and Scikit-learn.

## Tasks
1. **Data Preparation**:
   - Loaded the diabetes dataset using `sklearn.datasets`.
   - Split into training (80%) and test (20%) sets.
2. **From Scratch Implementation**:
   - Implemented Random Forest Regression from scratch using bagging and decision trees.
   - Tuned the number of trees using GridSearchCV.
   - Evaluated on the test set and reported MSE and OOB error.
3. **Scikit-learn Implementation**:
   - Used `RandomForestRegressor` from Scikit-learn.
   - Performed cross-validation to find the best number of trees.
   - Evaluated on the test set and reported MSE and OOB error.
4. **Comparison and Visualization**:
   - Compared MSE of both implementations.
   - Visualized predicted values vs. true values for both implementations.

## Results
- **Best Number of Trees**:
  - From Scratch: `<value>`
  - Scikit-learn: `<value>`
- **MSE**:
  - From Scratch: `<value>`
  - Scikit-learn: `<value>`

## How to Run
1. Install required libraries:
   ```bash
   pip install numpy scikit-learn matplotlib
