# Lab 8: Decision Trees and Support Vector Machines

## Overview

This laboratory assignment focuses on two of the most widely used supervised learning models in machine learning: decision trees and support vector machines (SVMs). The notebook implements a decision tree and a linear soft-margin SVM from scratch, compares those implementations against scikit-learn equivalents, and explores how kernel-based SVMs handle non-linear classification.

The work uses the UCI Banknote Authentication dataset, which contains 1,372 samples with four measured features derived from wavelet-transformed images of banknotes. The experiments are structured to compare decision boundaries, model performance, and generalisation on both a reduced two-feature view and the full four-feature classification problem.

## Project Structure

- `202618052_Lab_8.ipynb`: Complete implementation and analysis for the decision tree, linear SVM, kernel SVM, and final benchmark study.
- `data/data_banknote_authentication.csv`: Banknote authentication dataset used throughout the notebook.
- `figures/`: Generated plots and decision-boundary visualisations created during the analysis.
- `results/`: Saved summaries, model metrics, and experiment outputs.
- `Readme.md`: Documentation for this lab.

## Dataset

- **Dataset name:** UCI Banknote Authentication
- **Samples:** 1,372
- **Features:** 4 continuous attributes
  - Variance
  - Skewness
  - Curtosis
  - Entropy
- **Target:** Binary class label (0 or 1)
- **Task:** Classify genuine vs. forged banknotes

The original dataset file is stored locally in `data/data_banknote_authentication.csv`. The notebook reads it directly and performs a fixed stratified train-test split for consistent evaluation across all parts of the assignment.

## Key Components

### Part A: Decision Tree from Scratch

1. Implement a binary decision tree using impurity-based splitting.
2. Train the tree on a reduced two-feature problem using variance and skewness.
3. Visualise the learned decision boundary in 2D.
4. Compare the custom implementation with scikit-learn’s `DecisionTreeClassifier`.
5. Evaluate how tree depth and training accuracy relate to model complexity and possible overfitting.

### Part B: Linear Soft-Margin SVM from Scratch

1. Formulate a soft-margin linear SVM objective with regularisation.
2. Use subgradient descent to learn the separating hyperplane.
3. Plot the decision boundary for the two-feature subset.
4. Compare the custom implementation with `sklearn.svm.LinearSVC` and `SVC(kernel='linear')`.
5. Examine the effect of the regularisation parameter `C` on the margin and classification boundary.

### Part C: Kernel SVM and Non-Linear Boundaries

1. Extend the SVM to non-linear decision boundaries using kernels.
2. Evaluate polynomial and radial basis function (RBF) kernels.
3. Visualise non-linear boundaries on the 2D feature space.
4. Investigate the role of kernel parameters such as `gamma` in controlling model flexibility.
5. Compare the performance of kernel SVMs against the linear SVM.

### Part D: Final Comparison on the Full Banknote Dataset

1. Use all four input features for the final benchmarking stage.
2. Apply the same train-test split across all models to ensure comparability.
3. Compare:
   - Decision Tree
   - Linear SVM
   - RBF SVM
   - Other relevant scikit-learn baselines
4. Evaluate accuracy, precision, recall, F1-score, and model behaviour on the full feature space.
5. Summarise how the choice of model and representation influences performance.

## How to Run

### 1. Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

### 2. Open the notebook

From the repository root, start Jupyter in the lab directory:

```bash
cd 202618052_Lab_8
jupyter notebook 202618052_Lab_8.ipynb
```

Alternatively, open the notebook directly in VS Code with the Python and Jupyter extensions enabled and run the cells in order.

## Outputs

The notebook generates visual summaries and model-performance outputs, including:

- Decision boundary plots for decision trees and SVMs
- Margin and hyperplane visualisations
- Comparison charts for manual vs. scikit-learn implementations
- Metric summaries for both the two-feature and four-feature problems
- Saved figures in `figures/`
- Result tables and experiment outputs in `results/`

## Conclusion

Lab 8 demonstrates the practical differences between tree-based and margin-based classification methods. The decision tree offers an interpretable, rule-based structure, while the soft-margin SVM provides a powerful maximum-margin formulation that can be extended to non-linear classification through kernels. By comparing manual implementations with scikit-learn versions, the lab highlights both the theoretical foundations and the implementation considerations behind these widely used classifiers.

## Author

Lakshita Pagaria  
Registration Number: 202618052
