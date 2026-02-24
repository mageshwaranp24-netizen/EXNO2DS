# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```        
# ----------------------------------------
# Step 1: Import Required Packages
# ----------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------
# Step 2: Load the Dataset
# ----------------------------------------
data = pd.read_csv("Exp_2_dataset_titanic_dataset.csv")

print("\nDataset Loaded Successfully\n")
print(data.head())
print("\nDataset Info:\n")
print(data.info())
print(data.describe())
# ----------------------------------------
# Step 3: Data Cleansing - Handle Missing Values
# ----------------------------------------
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])   # Mode for categorical
    else:
        data[column] = data[column].fillna(data[column].median())   # Median for numerical

print("\nMissing values handled successfully.\n")

# ----------------------------------------
# Step 4: Boxplot to Analyze Outliers (Age & Fare)
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x=data["Age"])
plt.title("Boxplot - Age")
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x=data["Fare"])
plt.title("Boxplot - Fare")
plt.show()

# ----------------------------------------
# Step 5: Remove Outliers Using IQR Method
# ----------------------------------------
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

data = remove_outliers_iqr(data, "Age")
data = remove_outliers_iqr(data, "Fare")

print("Outliers removed using IQR method.\n")

# ----------------------------------------
# Step 6: Countplot for Categorical Data
# ----------------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=data)
plt.title("Countplot - Survival Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Sex", data=data)
plt.title("Countplot - Gender Distribution")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", data=data)
plt.title("Countplot - Passenger Class Distribution")
plt.show()

# ----------------------------------------
# Step 7: Displot for Univariate Distribution
# ----------------------------------------
sns.displot(data["Age"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Age Distribution")
plt.show()

sns.displot(data["Fare"], kde=True, height=4, aspect=1.5)
plt.title("Displot - Fare Distribution")
plt.show()

# ----------------------------------------
# Step 8: Cross Tabulation
# ----------------------------------------
print("\nCross Tabulation: Sex vs Survived\n")
print(pd.crosstab(data["Sex"], data["Survived"]))

print("\nCross Tabulation: Pclass vs Survived\n")
print(pd.crosstab(data["Pclass"], data["Survived"]))

# ----------------------------------------
# Step 9: Heatmap for Correlation Analysis
# ----------------------------------------
plt.figure(figsize=(8,6))
correlation_matrix = data.select_dtypes(include=np.number).corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap - Titanic Dataset")
plt.show()

```
OUTPUT:

<img width="1920" height="1080" alt="Screenshot (154)" src="https://github.com/user-attachments/assets/16453ac7-7d11-4bde-bc76-299730539ef0" />

<img width="1920" height="1080" alt="Screenshot (155)" src="https://github.com/user-attachments/assets/5f91e2c5-77f2-43f2-8326-6a8ba51a43b0" />

<img width="1920" height="1080" alt="Screenshot (156)" src="https://github.com/user-attachments/assets/ae2a8253-a3b8-4e11-a5e1-71cf8609260e" />

<img width="1920" height="1080" alt="Screenshot (157)" src="https://github.com/user-attachments/assets/f7c14b70-5544-4c18-81a4-9a1176ec1432" />

<img width="1920" height="1080" alt="Screenshot (158)" src="https://github.com/user-attachments/assets/f4fb663b-2e08-44d2-93cc-fe0c4e37012b" />

<img width="1920" height="1080" alt="Screenshot (159)" src="https://github.com/user-attachments/assets/eaf77692-ee37-4974-acd5-23113fd62fca" />

<img width="1920" height="1080" alt="Screenshot (160)" src="https://github.com/user-attachments/assets/c9fb81a2-3eb2-437d-9423-29ac4210e313" />

<img width="1920" height="1080" alt="Screenshot (161)" src="https://github.com/user-attachments/assets/d21bb6e4-3532-4feb-8773-7b9d8290b921" />

<img width="1920" height="1080" alt="Screenshot (162)" src="https://github.com/user-attachments/assets/00d00638-5d10-416d-9969-077340ed154c" />

<img width="1920" height="1080" alt="Screenshot (163)" src="https://github.com/user-attachments/assets/6bf575f1-c251-49f1-be68-05b754ccb98f" />



# RESULT

       Thus the  Exploratory Data Analysis on the given data set was successfully completed. 
