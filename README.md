# 💼 Bank-Customer-Churn-Prediction

## 📌 Project Overview & Task Objective

The`Customer_Churn_Prediction.ipynb` focused on predicting customer churn for a bank.
The primary objective is to build classification models that can accurately identify 
customers who are likely to leave the bank, leveraging the provided **Churn Modelling Dataset**.

## 📂 Dataset Information

The project utilizes the **Churn Modelling Dataset** which contains features related to bank customers.  
The **target variable** is `Exited` (`1 = churned`, `0 = not churned`).
**Key Issues Handled:**
- Conversion of categorical data (e.g., `Geography`, `Gender`) to numerical format.

## ✨ Features

- Data loading and initial inspection  
- Handling categorical variables through one-hot encoding  
- Exploratory Data Analysis (EDA) to understand feature distributions  
- Model training and evaluation:
  - Random Forest Classifier
  - Logistic Regression
  - Gradient Boosting Classifier  

## 🛠️ Installation

To run this notebook locally, install the required libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib
```

## 🚀 Approach
My approach to customer churn prediction involved the following steps:
### 1. Library Import  
Used libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`.

### 2. Data Loading  
Loaded `Churn_Modelling.csv` into a pandas DataFrame.

### 3. Data Cleaning & Preparation  
- Dropped irrelevant columns: `RowNumber`, `CustomerId`, `Surname`  
- Encoded `Geography` and `Gender` using one-hot encoding

### 4. Exploratory Data Analysis (EDA)  
Visualized feature distributions and relationships with the target variable (`Exited`).

### 5. Model Training & Testing  
The dataset was split into training and testing sets (typically 80/20 split).
- Random Forest Classifier: A Random Forest model was trained.
- Logistic Regression: A Logistic Regression model was trained for binary classification.
- Gradient Boosting Classifier: A Gradient Boosting model was trained.
- 
### 6. Model Evaluation  
Evaluated the trained models using :
- `Accuracy score`, `classification reports`,
  `confusion matrices`, and `ROC curves` to assess their performance in predicting
   customer churn.

## 🧰 Technologies Used

-  Python  
-  Pandas  
-  NumPy  
-  Matplotlib  
-  Seaborn  
-  Scikit-learn  
-  Joblib  

## 📉 Visualizations

### 📊 Churn Distribution  
![image](https://github.com/user-attachments/assets/92e01180-a4ff-4ba4-b5f4-d2d282073727)

**Insight**: 
The Above Graph Shows that:
The dataset is imbalanced:
- Majority (~80%) of customers did not churn.
- Minority (~20%) exited.

### 📈 Feature Distributions  
**Age Distribution**
![image](https://github.com/user-attachments/assets/14efb244-5f20-43f1-b731-5f4690d7de88)

**Insight**:
The Above Graph Shows that:
- Most customers are between 30 and 45 years old.
- Very few are under 20 or over 60.
The distribution is slightly right-skewed.
Age could play a role in customer churn.
**Tenure Vs Churn**
![image](https://github.com/user-attachments/assets/f71f902f-e6c9-4bf0-9572-8b50c440e51d)

**Insight**:
The Above Grapsh Shows that:
Churn is fairly uniform across all tenure levels (0–10 years).
- No clear trend of more or less churn at any specific tenure.
- Tenure alone may not be a strong predictor.
### 🟢 Feature Importance
![image](https://github.com/user-attachments/assets/bdfdb258-e773-478d-a742-d81018ef1b63)

**Insights**:
The Above Graph evaluate the Importance of Feature
Top features influencing churn:
- `Age` is the most influential feature in this model.
- Other strong predictors include `EstimatedSalary`, `CreditScore`, and `Balance` and `Number of Products`.
- Features like `Geography_Spain`, `Gender_Male`, and `HasCrCard` have much lower importance These features are key drivers in the Random Forest model’s decisions.

### 📈 ROC Curve  
![image](https://github.com/user-attachments/assets/b2ce25c5-46be-413e-80ff-47da4b76d2d9)

**Insights**:
ROC Curve – Random Forest
Displays model performance across different thresholds.  
**ROC** curve shows the trade-off between true positive rate and false positive rate.
**AUC** score closer to 1 means better model performance.
This Random Forest model has a good AUC score, indicating strong predictive power.

## 📊 Results and Insights

### 🔎 Key Insights:

- **Churn Rate**: Identified the proportion of churned customers.  
- **Feature Importance**: Variables like `Age`, `Balance`, and `NumOfProducts` played crucial roles.  
- **Model Performance**:
  - Random Forest: `86.65%`
  - Gradient Boosting: `86.75%` (slightly best)
  - Logistic Regression: `81.10%` (lowest)
  - Ensemble models (Random Forest, Gradient Boosting) outperformed Logistic Regression.
  - Achieved higher accuracy and F1-scores.  

### 📉 Evaluation Metrics:

- **Confusion Matrices** and **ROC Curves** provided insight into:
  - True/False Positives & Negatives
  - Classifier threshold performance

### ✅ Final Outcome:
This project demonstrated a complete end-to-end churn prediction workflow.  
It can help banks proactively identify at-risk customers.  
Further improvements can include:
- Hyperparameter tuning  
- Advanced feature engineering  
- Exploring deep learning approaches

## 🧪 Usage

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Customer-Churn-Prediction.git

# 2. Navigate to the project directory
cd Customer-Churn-Prediction

# 3. Open the notebook
jupyter notebook Customer_Churn_Prediction.ipynb

# 4. Run the notebook cells
```

## 🤝 Contributing

Contributions are welcome!  
Please open an issue or submit a pull request for suggestions or improvements.

## 📬 Contact

For questions or collaboration:

- GitHub: `AhsanNFt`
- Email: `syedahsan0991@gmail.com`.
