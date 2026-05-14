# CustomGBM: High-Performance Gradient Boosting & Random Forest From Scratch

An end-to-end, zero-dependency machine learning project implementing a custom **Gradient Boosting Regressor**, **Random Forest Regressor**, and an automated data preprocessing **Pipeline** from scratch using only `NumPy` and `Pandas`. 

The core task of this repository is to accurately predict the **mean math exam points** scored by students based on their tutors' profiles and professional characteristics.

---

## 🎯 Project Overview & Objectives

Predicting student performance based on tutor qualifications is a highly non-linear regression problem riddled with complex interactions (e.g., the relationship between price, experience, and educational background). While standard libraries like `scikit-learn` or `XGBoost` handle this easily, this project focuses on **algorithmic transparency and mastery**, building the entire architecture from basic mathematical foundations.

### Core Objectives:
* Implement a robust **Decision Tree Regressor** utilizing Mean Squared Error (MSE) minimization.
* Construct an ensemble **Random Forest** algorithm leveraging bagging and feature subspaces.
* Develop a **Gradient Boosting Machine (GBM)** utilizing additive modeling, custom residual mapping, and integrated state-rollback.
* Incorporate production-grade optimizations like **Early Stopping** and cumulative **Feature Importance tracking**.

---

## 🧬 Data Pipeline & Feature Engineering

Raw data contains implicit signals that individual trees can struggle to parse efficiently. To maximize model performance, a custom `Pipeline` class was built to handle feature mutations globally without causing **Data Leakage**.

### Raw Features:
* `Id`: Unique identifier
* `age`: Tutor's age
* `years_of_experience`: Years teaching
* `lesson_price`: Cost per lesson
* `qualification`: Internal rating of the tutor's background
* `physics`, `chemistry`, `biology`, `english`, `geography`, `history`: Binary flags indicating subjects taught

### Custom Feature Transformations:
1.  **Categorical Mapping (`modifying`)**: Outlier correction and targeted target-encoding alignments for specialized metrics (e.g., standardizing `qualification` boundaries).
2.  **Price-to-Quality Ratio (`divide`)**: Creates the `price_per_qual` feature to highlight cost-efficiency:
    $$\text{price per qual}=\frac{\text{lesson price}}{\text{qualification}}$$
3.  **Efficiency Index (`effective_index`)**: Combines compounding experience with structural background cost metrics into a unified interaction vector:
    $$\text{experience qval price} = \frac{\text{qualification} \cdot (\text{years of experience} + 1)}{\text{lesson price}}$$
4.  **Feature Pruning**: Automated dropping of highly collinear source variables (`lesson_price`, `years_of_experience`) to streamline information density.

---

## 🛠 Model Architecture & Mathematical Foundations

### 1. Gradient Boosting Regression
The model fits consecutive trees to the negative gradient of the Mean Squared Error (MSE) loss function.

* **Loss Function**: 
    $$L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2$$
* **Pseudo-Residuals Calculation**: For each training sample, the residual is computed as the negative gradient of the loss with respect to the current ensemble prediction:
    $$r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right] {f(x)=f_{m-1}(x)} = y_i - f_{m-1}(x_i)$$
* **Additive Updates**:
    $$f_m(x) = f_{m-1}(x) + \eta \cdot h_m(x)$$
    (Where $\eta$ represents the learning rate `eta`, and $h_m(x)$ is the newly trained base tree).

### 2. Advanced Algorithmic Enhancements
* **Early Stopping with State Rollback**: To eliminate overfitting and reduce redundant computations, training monitors validation error over an evaluation sequence (`early_stopping_rounds`). If no metric improvements are detected, the ensemble terminates training and actively **rolls back** its internal arrays (`self.trees`, `self.total_feature_importances_`) to the historical optimum state.
* **Ensemble Feature Importance**: Tracks structural variance reduction across every split across every tree, weighted by the sample distribution passing through the parent node.

$$\text{Gain} = \text{MSE(root)} - \left( p_{\text{left}} \cdot \text{MSE(left)} + p_{\text{right}} \cdot \text{MSE(right)} \right)$$
---

## 📊 Performance Benchmarks

The custom Gradient Boosting Machine was heavily benchmarked against the custom Random Forest implementation on a fully isolated holdout **Test Set**. 

### Empirical Evaluation:

| Model Hierarchy | Dataset Split | MAE | MSE | RMSE | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest** | Train | 6.7393 | 69.8253 | 8.3562 | 0.6825 |
| | Valid | 6.8151 | 71.5740 | 8.4601 | 0.6573 |
| | **Test** | **7.0416** | **76.1742** | **8.7278** | **0.6411** |
| 📊 **Custom GBM** | Train | 6.7792 | 70.7158 | 8.4093 | 0.6784 |
| *(Optimal)* | Valid | 6.7290 | 69.5622 | 8.3404 | 0.6670 |
| | **Test** | **6.9025** | **72.5168** | **8.5157** | **0.6583** |


<img width="425" height="415" alt="image" src="https://github.com/user-attachments/assets/8143a90d-5987-4c08-919e-0495c70b3a77" />


### Key Takeaways:
* **Generalization Error Drop**: The Gradient Boosting model significantly lowered test variance. Test MSE dropped from **76.17** down to **72.51** (a **~4.8% reduction in squared error**).
* **Variance Control**: The gap between Validation and Test metrics is highly compact in GBM, demonstrating that the combination of shallow base estimators (`max_depth=3`) and automated Early Stopping successfully neutralized overfitting.

---

## 🧬 Feature Importance Insights

The computed cumulative importance vector exposes exactly what variables steer the predictions of the ensemble:

```markdown
  Variable                  Relative Variance Reduction Score
  -----------------------------------------------------------
  qualification             █████████████████████████ 589.57
  price_per_qual            ████ 99.56
  experience_qval_price     ██ 52.83
  physics                   █ 35.96
  age                       ▏ 3.75
  biology                   ▏ 1.42
  chemistry                 ▏ 1.32
  english                   ▏ 0.51
  geography                 ▏ 0.33
  history                     0.00
