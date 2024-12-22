# import necessory libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

"""# **Task-1**"""

df = sklearn.datasets.load_diabetes(return_X_y=False, as_frame=False, scaled=True)

display(df)

X = df.data
y = df.target

data = pd.DataFrame(data=np.c_[X, y], columns=df.feature_names + ['target'])

display(data)

data.describe()

data.isnull().sum()

data.duplicated().sum()

data.isna().sum()

#  Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Split the dataset and ensure arrays are used for compatibility with numpy indexing
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# using simple imputer to avoid NAN values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standerdize the data set

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)

"""# **Task-2**"""

class decisiontree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth == self.max_depth or n_samples < self.min_samples_split:
            return np.mean(y)

        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None or best_threshold is None:
            return np.mean(y)

        left_mask = X[:, best_feature] < best_threshold
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return (best_feature, best_threshold, left, right)

    def _find_best_split(self, X, y):
        best_feature, best_threshold, best_mse = None, None, float('inf')
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    mse = (np.sum((y[left_mask] - np.mean(y[left_mask]))**2) +
                           np.sum((y[right_mask] - np.mean(y[right_mask]))**2))
                    if mse < best_mse:
                        best_feature, best_threshold, best_mse = feature, threshold, mse
        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        node = self.tree
        while isinstance(node, tuple):
            if sample[node[0]] < node[1]:
                node = node[2]
            else:
                node = node[3]
        return node


class randomforest:
    def __init__(self, n_trees=100, max_features='sqrt', max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.oob_indices = []

    def _sample_features(self, X):
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        selected_features = np.random.choice(n_features, max_features, replace=False)
        return selected_features

    def fit(self, X, y):
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = [i for i in range(n_samples) if i not in indices]
            self.oob_indices.append(oob_indices)

            X_sample, y_sample = X[indices], y[indices]
            selected_features = self._sample_features(X)
            X_sample = X_sample[:, selected_features]

            tree = decisiontree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, selected_features))

    def predict(self, X):
        tree_preds = np.zeros((self.n_trees, X.shape[0]))
        for i, (tree, features) in enumerate(self.trees):
            X_subset = X[:, features]
            tree_preds[i] = tree.predict(X_subset)
        return np.mean(tree_preds, axis=0)

    def oob_score(self, X, y):
        n_samples = X.shape[0]
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for i, (tree, features) in enumerate(self.trees):
            oob_indices = self.oob_indices[i]
            if oob_indices:
                X_oob = X[oob_indices][:, features]
                oob_preds = tree.predict(X_oob)
                oob_predictions[oob_indices] += oob_preds
                oob_counts[oob_indices] += 1

        # Only calculate MSE for samples that have OOB predictions
        mask = oob_counts > 0
        oob_predictions[mask] /= oob_counts[mask]
        oob_mse = mean_squared_error(y[mask], oob_predictions[mask])

        return oob_mse

"""#**Task-3**"""

# Set up parameter grid
param_grid = {'n_trees': [10, 50, 100, 200]}

# Define a custom grid search function or wrap the model in a GridSearchCV-compatible structure if possible.
best_mse = float("inf")
best_n_trees = 0

for n_trees in param_grid['n_trees']:
    rf = randomforest(n_trees=n_trees)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    oob_error = rf.oob_score(X_train, y_train)

    if mse < best_mse:
        best_mse = mse
        best_n_trees = n_trees

print(f"Best number of trees: {best_n_trees}")
print(f"MSE: {best_mse}, OOB Error: {oob_error}")

import matplotlib.pyplot as plt

# Define a function to plot a decision tree
def plot_decision_tree(tree, feature_names, depth=0):
    if isinstance(tree, tuple):
        idx, thr, left, right = tree
        feature_name = feature_names[idx]
        # Plot decision node
        print('  ' * depth + f'if {feature_name} <= {thr}:')
        plot_decision_tree(left, feature_names, depth + 1)
        print('  ' * depth + f'else:')
        plot_decision_tree(right, feature_names, depth + 1)
    else:
        # Leaf node
        print('  ' * depth + f'class: {tree}')

# Select a few decision trees for visualization
trees_to_visualize = [0, 1, 2, 3, 4]  # You can adjust this list as needed

for i in trees_to_visualize:
    print(f"Decision Tree {i+1}:")
    # Access the decision tree object from the tuple using index 0
    plot_decision_tree(rf.trees[i][0].tree, [f'Feature {i}' for i in range(20)])
    print()



"""# **Task-4**"""

param_grid = {'n_estimators': [10, 50, 100, 200]}
rf_model = RandomForestRegressor(oob_score=True, random_state=42)

grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
best_mse = mean_squared_error(y_test, best_rf.predict(X_test))
oob_error = best_rf.oob_score_

print(f"Optimal n_estimators: {grid_search.best_params_['n_estimators']}")
print(f"MSE: {best_mse}, OOB Error: {oob_error}")

"""# **Task-5**"""

import matplotlib.pyplot as plt

# Predictions from scratch model
rf_scratch = randomforest(n_trees=best_n_trees)
rf_scratch.fit(X_train, y_train)
y_pred_scratch = rf_scratch.predict(X_test)

# Predictions from scikit-learn model
y_pred_sklearn = best_rf.predict(X_test)

# Scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for the from-scratch implementation
axes[0].scatter(y_test, y_pred_scratch, alpha=0.6, color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[0].set_title("From-Scratch Implementation")
axes[0].set_xlabel("True Values")
axes[0].set_ylabel("Predicted Values")

# Plot for the sklearn implementation
axes[1].scatter(y_test, y_pred_sklearn, alpha=0.6, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
axes[1].set_title("Scikit-learn Implementation")
axes[1].set_xlabel("True Values")
axes[1].set_ylabel("Predicted Values")

plt.tight_layout()
plt.show()
