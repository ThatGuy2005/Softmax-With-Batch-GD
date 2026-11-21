# Linear & Softmax Regression from Scratch

A first-principles NumPy implementation of **Batch Gradient Descent** for both Ordinary Least Squares (OLS) linear regression and multi-class **Softmax Regression**, built without high-level ML model APIs.

## Key Implementation Details
* **Vectorized Optimization:** Implemented batch gradient updates from scratch using matrix operations (`X_b.T @ (P - one_hot_y)`).
* **Cross-Entropy Loss & Early Stopping:** Added validation-loss monitoring (`-mean(sum(Y * log(P + eps)))`) with configurable `patience` to halt training before overfitting and restore the lowest-loss weight matrix (`best_theta`).
* **Stratified Data Pipeline:** Split the Iris dataset (`petal length`, `petal width`) into stratified 60% train / 20% validation / 20% test sets (`random_state=42`).

## Results (Iris Dataset)
* **Hyperparameters:** Learning rate $\eta = 0.1$, max epochs $= 1{,}000{,}000$, early stopping patience $= 120$
* **Convergence:** Stopped around epoch ~170,000 at a validation cross-entropy loss of **0.2022**
* **Accuracy:** **96.0%** classification accuracy

## Project Structure
* `*.ipynb` — Contains:
  1. Synthetic linear regression testbed (`y = 4 + 3x + noise`) verifying gradient descent parameter recovery (`[4.22, 2.77]`).
  2. Multi-class Softmax regression with early stopping evaluated on the Iris dataset.

## Dependencies
```bash
pip install numpy matplotlib scikit-learn
