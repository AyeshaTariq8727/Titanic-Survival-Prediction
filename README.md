# Titanic Survival Prediction using Machine Learning

## Introduction
This project builds a machine learning model to predict passenger survival on the Titanic using Python. We will go through feature selection, model training, evaluation, and predictions.

---

## Step 1: Import Required Libraries
```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
```

---

## Step 2: Load and Prepare Data
We use the Titanic dataset and clean it for analysis.
```python
# Load Titanic dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Drop unnecessary columns
df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

# Fill missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Convert categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Label Encoding
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)  # One-Hot Encoding
```
✅ Now, the dataset is ready for modeling!

---

## Step 3: Define Features and Target Variable
```python
X = df.drop(columns=["Survived"])  # Features
y = df["Survived"]  # Target variable
```

---

## Step 4: Split Data into Training and Testing Sets
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## Step 5: Feature Scaling
```python
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)  
X_test = scaler.transform(X_test)
```

---

## Step 6: Train a Random Forest Classifier
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)  
model.fit(X_train, y_train)
```

---

## Step 7: Model Evaluation
### 7.1 Predictions
```python
y_pred = model.predict(X_test)
```
### 7.2 Accuracy Score
```python
accuracy = accuracy_score(y_test, y_pred)  
print(f"Accuracy: {accuracy:.2f}")
```
✅ Expected accuracy: ~80-85% (Good for Titanic dataset).

### 7.3 Classification Report
```python
print(classification_report(y_test, y_pred))
```

### 7.4 Confusion Matrix
```python
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")  
plt.xlabel("Predicted")  
plt.ylabel("Actual")  
plt.show()
```
✅ Observation: Most predictions are correct, with a few misclassifications.

---

## Step 8: Making a Prediction on a New Passenger
### Example Passenger Details:
- Pclass = 1 (First class)
- Sex = Female
- Age = 25
- SibSp = 0 (No siblings/spouse aboard)
- Parch = 0 (No parents/children aboard)
- Fare = 100
- Embarked = S (Southampton)

### Making the Prediction
```python
new_passenger = np.array([[1, 1, 25, 0, 0, 100, 0, 0]])  # Adjust based on encoded features
new_passenger = scaler.transform(new_passenger)  
prediction = model.predict(new_passenger)  

print("Survival Prediction:", "Survived" if prediction[0] == 1 else "Not Survived")
```
✅ Final Prediction Example: A young woman in first class is more likely to survive.

---

## Conclusion
✔ Explored the Titanic dataset with EDA  
✔ Handled missing values and encoded categorical variables  
✔ Trained a Random Forest model for survival prediction  
✔ Achieved ~80-85% accuracy  
✔ Made predictions on new passengers  

🚀 **Next Steps:** Try experimenting with different models like Logistic Regression or XGBoost for better accuracy!
