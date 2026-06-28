# E-commerce Sales Analysis & Forecasting
# Author: Vinod M | mopurivinod6788@gmail.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# =====================
# 1. CREATE SAMPLE DATA
# =====================
np.random.seed(42)
n = 50000

data = {
    'order_id': range(1, n+1),
    'customer_id': np.random.randint(1000, 9999, n),
    'age': np.random.randint(18, 65, n),
    'gender': np.random.choice(['Male', 'Female'], n),
    'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], n),
    'quantity': np.random.randint(1, 10, n),
    'unit_price': np.round(np.random.uniform(10, 500, n), 2),
    'discount': np.round(np.random.uniform(0, 0.5, n), 2),
    'payment_method': np.random.choice(['Credit Card', 'UPI', 'Net Banking', 'COD'], n),
    'city': np.random.choice(['Hyderabad', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai'], n),
    'churned': np.random.choice([0, 1], n, p=[0.75, 0.25])
}

df = pd.DataFrame(data)
df['total_price'] = df['quantity'] * df['unit_price'] * (1 - df['discount'])
df['total_price'] = df['total_price'].round(2)

# =====================
# 2. DATA CLEANING
# =====================
print("=== DATA OVERVIEW ===")
print(f"Total Records: {len(df)}")
print(f"Missing Values:\n{df.isnull().sum()}")
print(f"\nData Types:\n{df.dtypes}")

# =====================
# 3. SALES ANALYSIS
# =====================
print("\n=== SALES ANALYSIS ===")
print(f"Total Revenue: ₹{df['total_price'].sum():,.2f}")
print(f"Average Order Value: ₹{df['total_price'].mean():,.2f}")
print(f"Top Category by Revenue:\n{df.groupby('category')['total_price'].sum().sort_values(ascending=False)}")

# =====================
# 4. VISUALIZATIONS
# =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('E-commerce Sales Analysis - Vinod M', fontsize=16, fontweight='bold')

# Chart 1: Revenue by Category
category_revenue = df.groupby('category')['total_price'].sum().sort_values(ascending=False)
axes[0, 0].bar(category_revenue.index, category_revenue.values, color=['#2563EB','#0EA5E9','#10B981','#F59E0B','#EF4444'])
axes[0, 0].set_title('Revenue by Category')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Total Revenue (₹)')
axes[0, 0].tick_params(axis='x', rotation=15)

# Chart 2: Revenue by City
city_revenue = df.groupby('city')['total_price'].sum().sort_values(ascending=False)
axes[0, 1].barh(city_revenue.index, city_revenue.values, color='#2563EB')
axes[0, 1].set_title('Revenue by City')
axes[0, 1].set_xlabel('Total Revenue (₹)')

# Chart 3: Payment Method Distribution
payment_counts = df['payment_method'].value_counts()
axes[1, 0].pie(payment_counts.values, labels=payment_counts.index, autopct='%1.1f%%',
               colors=['#2563EB','#0EA5E9','#10B981','#F59E0B'])
axes[1, 0].set_title('Payment Method Distribution')

# Chart 4: Churn Distribution
churn_counts = df['churned'].value_counts()
axes[1, 1].bar(['Active', 'Churned'], churn_counts.values, color=['#10B981', '#EF4444'])
axes[1, 1].set_title('Customer Churn Distribution')
axes[1, 1].set_ylabel('Number of Customers')

plt.tight_layout()
plt.savefig('sales_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved as sales_analysis.png")

# =====================
# 5. CHURN PREDICTION ML MODEL
# =====================
print("\n=== CHURN PREDICTION MODEL ===")

le = LabelEncoder()
df['gender_enc'] = le.fit_transform(df['gender'])
df['category_enc'] = le.fit_transform(df['category'])
df['payment_enc'] = le.fit_transform(df['payment_method'])
df['city_enc'] = le.fit_transform(df['city'])

features = ['age', 'gender_enc', 'category_enc', 'quantity',
            'unit_price', 'discount', 'payment_enc', 'city_enc', 'total_price']
X = df[features]
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Model Accuracy: {model.score(X_test, y_test)*100:.2f}%")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Feature Importance
feat_importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(f"\nTop Features:\n{feat_importance}")

print("\n✅ Analysis Complete! Vinod M - Data Analyst Portfolio Project")