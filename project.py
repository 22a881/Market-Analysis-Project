
# ==========================================
# مشروع تحليل أسعار السوق وكشف الأزمات
# المرحلة 0: إنشاء البيانات
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    silhouette_score
)
# تثبيت العشوائية للحصول على نفس البيانات
np.random.seed(42)

# إنشاء مجلد البيانات إذا لم يكن موجوداً
Path("data").mkdir(exist_ok=True)

# أسماء السلع
items = [
    "Flour_25kg",
    "Sugar",
    "Rice",
    "Tomato",
    "Potato",
    "Oil"
]

# الأسعار الأساسية للسلع بالشيكل
base_prices = {
    "Flour_25kg": 120,
    "Sugar": 6,
    "Rice": 10,
    "Tomato": 5,
    "Potato": 4,
    "Oil": 18
}

# إنشاء تواريخ 90 يوماً
dates = pd.date_range(
    start="2024-01-01",
    periods=90,
    freq="D"
)

# قائمة لتجميع البيانات
data = []

# المرور على كل يوم
for day_index, date in enumerate(dates):

    # سعر الدولار
    usd_rate = 3.65 + np.random.normal(0, 0.05)

    # حالة المعبر
    border_open = np.random.choice(
        [0, 1],
        p=[0.2, 0.8]
    )

    # مستوى الأزمة
    if day_index < 30:
        crisis_level = np.random.choice(
            [0, 1, 2],
            p=[0.8, 0.15, 0.05]
        )

    elif day_index < 60:
        crisis_level = np.random.choice(
            [0, 1, 2],
            p=[0.3, 0.5, 0.2]
        )

    else:
        crisis_level = np.random.choice(
            [0, 1, 2],
            p=[0.1, 0.3, 0.6]
        )

    # المرور على الأصناف الستة
    for item in items:

        base_price = base_prices[item]

        # تأثير الدولار
        dollar_effect = (usd_rate - 3.65) * base_price * 0.5

        # تأثير إغلاق المعبر
        border_effect = 0

        if border_open == 0:
            border_effect = base_price * 0.15

        # تأثير الأزمة
        crisis_effect = crisis_level * base_price * 0.12

        # تغير عشوائي بسيط
        random_effect = np.random.normal(
            0,
            base_price * 0.04
        )

        # حساب السعر النهائي
        price = (
            base_price
            + dollar_effect
            + border_effect
            + crisis_effect
            + random_effect
        )

        # منع الأسعار السالبة
        price = max(price, 0.1)

        # تحديد حالة التوفر
        if crisis_level == 2 and border_open == 0:
            availability = "شحيح"

        elif crisis_level >= 1 or border_open == 0:
            availability = "ناقص"

        else:
            availability = "متوفر"

        # إضافة السجل
        data.append({
            "date": date,
            "item": item,
            "price_shekel": round(price, 2),
            "usd_rate": round(usd_rate, 3),
            "border_open": border_open,
            "crisis_level": crisis_level,
            "availability": availability
        })


# تحويل البيانات إلى DataFrame
df = pd.DataFrame(data)

# حفظ البيانات في ملف CSV
df.to_csv(
    "data/market_data.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print("MARKET DATA GENERATED SUCCESSFULLY")
print("===================================")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nDataset Information:")
df.info()

print("\nStatistical Description:")
print(df.describe(include="all"))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCSV file saved successfully!")

# ==========================================
# المرحلة 0: التحليل الإحصائي التفصيلي
# ==========================================

print("\n" + "=" * 60)
print("PHASE 0: STATISTICAL ANALYSIS")
print("=" * 60)

# 1. الإحصائيات لكل صنف
print("\n1. Statistical Analysis By Item")

item_stats = df.groupby("item")["price_shekel"].agg(
    ["mean", "median", "std", "min", "max"]
)

item_stats.columns = [
    "Mean",
    "Median",
    "Standard Deviation",
    "Minimum",
    "Maximum"
]

print(item_stats.round(2))


# 2. متوسط الأسعار حسب حالة المعبر
print("\n2. Average Price By Border Status")

border_stats = df.groupby("border_open")["price_shekel"].mean()

print(border_stats)

print("0 = Border Closed")
print("1 = Border Open")


# 3. متوسط الأسعار حسب مستوى الأزمة
print("\n3. Average Price By Crisis Level")

crisis_stats = df.groupby("crisis_level")["price_shekel"].mean()

print(crisis_stats)

print("0 = Calm")
print("1 = Moderate Crisis")
print("2 = Severe Crisis")


# 4. Pivot Table
print("\n4. Pivot Table: Border × Crisis")

pivot_table = pd.pivot_table(
    df,
    values="price_shekel",
    index="border_open",
    columns="crisis_level",
    aggfunc="mean"
)

print(pivot_table.round(2))


# 5. توزيع حالات التوفر بالنسب المئوية
print("\n5. Availability Distribution (%)")

availability_percent = (
    df["availability"]
    .value_counts(normalize=True) * 100
)

print(availability_percent.round(2))


# حفظ نتائج التحليل
item_stats.to_csv("data/item_statistics.csv", encoding="utf-8-sig")

pivot_table.to_csv("data/pivot_table.csv", encoding="utf-8-sig")

print("\nStatistical Analysis Completed Successfully!")

# ==========================================
# المرحلة 1: K-Means Clustering
# تجميع أيام السوق إلى ثلاث حالات
# ==========================================

print("\n" + "=" * 60)
print("PHASE 1: K-MEANS CLUSTERING")
print("=" * 60)

# 1. إنشاء Pivot Table
price_pivot = df.pivot_table(
    index="date",
    columns="item",
    values="price_shekel",
    aggfunc="mean"
)

print("\n1. Price Pivot Table:")
print(price_pivot.head())


# 2. تطبيع الأسعار باستخدام الوسيط
item_medians = price_pivot.median()

normalized_prices = price_pivot / item_medians

print("\n2. Normalized Prices:")
print(normalized_prices.head())


# 3. حساب مؤشر الأسعار اليومي
market_daily = pd.DataFrame(index=price_pivot.index)

market_daily["price_index"] = normalized_prices.mean(axis=1)

print("\n3. Daily Price Index:")
print(market_daily.head())


# 4. تطبيق StandardScaler
scaler_kmeans = StandardScaler()

price_scaled = scaler_kmeans.fit_transform(
    market_daily[["price_index"]]
)


# 5. اختبار Elbow Method
inertias = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(price_scaled)

    inertias.append(model.inertia_)

print("\n5. Elbow Method Results:")

for k, inertia in zip(range(2, 7), inertias):
    print(f"K = {k} | Inertia = {inertia:.4f}")


# رسم اختبار الكوع
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 7),
    inertias,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(range(2, 7))
plt.grid(True)

plt.savefig("figures/elbow_method.png", dpi=300)
plt.show()


# 6. تطبيق KMeans بثلاث مجموعات
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

market_daily["cluster"] = kmeans.fit_predict(price_scaled)


# 7. ترتيب المجموعات حسب متوسط السعر
cluster_means = market_daily.groupby("cluster")["price_index"].mean()

sorted_clusters = cluster_means.sort_values().index.tolist()

state_names = [
    "رخاء",
    "غلاء متوسط",
    "أزمة"
]

cluster_mapping = {
    cluster: state_names[index]
    for index, cluster in enumerate(sorted_clusters)
}

market_daily["state"] = market_daily["cluster"].map(cluster_mapping)

print("\n7. Cluster Mapping:")
print(cluster_mapping)


# 8. ملخص المجموعات
print("\n8. Cluster Summary:")

cluster_summary = market_daily.groupby("state").agg(
    number_of_days=("price_index", "count"),
    average_price_index=("price_index", "mean")
)

print(cluster_summary.round(4))


# 9. تقييم جودة التجميع
silhouette = silhouette_score(
    price_scaled,
    market_daily["cluster"]
)

print("\n9. Clustering Evaluation:")

print(f"Silhouette Score: {silhouette:.4f}")
print(f"Inertia: {kmeans.inertia_:.4f}")


# 10. دمج حالة السوق مع البيانات الأصلية
market_conditions = (
    df.groupby("date")
    .agg(
        border_open=("border_open", "first"),
        crisis_level=("crisis_level", "first")
    )
)

market_daily = market_daily.join(market_conditions)

df = df.merge(
    market_daily[["state"]],
    left_on="date",
    right_index=True,
    how="left"
)

print("\n10. Market State Analysis:")

state_analysis = df.groupby("state").agg(
    average_border_open=("border_open", "mean"),
    average_crisis_level=("crisis_level", "mean"),
    average_price=("price_shekel", "mean")
)

print(state_analysis.round(4))


# النسب المئوية للإغلاق والأزمة الشديدة داخل كل حالة
daily_analysis = market_daily.groupby("state").agg(
    border_closed_percentage=(
        "border_open",
        lambda x: (x == 0).mean() * 100
    ),
    severe_crisis_percentage=(
        "crisis_level",
        lambda x: (x == 2).mean() * 100
    )
)

print("\nCrisis and Border Closure Percentages:")
print(daily_analysis.round(2))


# حفظ نتائج التجميع
market_daily.to_csv(
    "data/market_clustering.csv",
    encoding="utf-8-sig"
)

cluster_summary.to_csv(
    "data/cluster_summary.csv",
    encoding="utf-8-sig"
)


# رسم حالات السوق
plt.figure(figsize=(12, 6))

sns.scatterplot(
    data=market_daily.reset_index(),
    x="date",
    y="price_index",
    hue="state",
    palette="viridis",
    s=65
)

plt.title("Market States Using K-Means")
plt.xlabel("Date")
plt.ylabel("Price Index")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("figures/market_clusters.png", dpi=300)
plt.show()

print("\nK-Means Clustering Completed Successfully!")

# ==========================================
# PHASE 2: ANOMALY DETECTION
# كشف القيم الشاذة في بيانات السوق
# ==========================================

print("\n" + "=" * 60)
print("PHASE 2: ANOMALY DETECTION")
print("=" * 60)

# 1. تجهيز البيانات
anomaly_data = df.copy()

# تحويل التوفر إلى أرقام
availability_mapping = {
    "متوفر": 2,
    "ناقص": 1,
    "شحيح": 0
}

anomaly_data["avail_num"] = (
    anomaly_data["availability"].map(availability_mapping)
)

# تحديد الخصائص المستخدمة
anomaly_features = [
    "price_shekel",
    "usd_rate",
    "border_open",
    "crisis_level",
    "avail_num"
]

X_anomaly = anomaly_data[anomaly_features]

# 2. توحيد المقاييس
scaler_anomaly = StandardScaler()
X_scaled = scaler_anomaly.fit_transform(X_anomaly)

# 3. حساب مركز البيانات
center = X_scaled.mean(axis=0)

# 4. حساب المسافة الإقليدية
distances = np.linalg.norm(
    X_scaled - center,
    axis=1
)

anomaly_data["distance"] = distances

# 5. تحديد الحد الفاصل لأعلى 5%
threshold = np.percentile(distances, 95)

anomaly_data["anomaly"] = np.where(
    anomaly_data["distance"] >= threshold,
    -1,
    1
)

# 6. عرض النتائج
anomaly_count = (anomaly_data["anomaly"] == -1).sum()
normal_count = (anomaly_data["anomaly"] == 1).sum()

print("\nAnomaly Detection Results:")
print(f"Total Records: {len(anomaly_data)}")
print(f"Anomalies: {anomaly_count}")
print(f"Normal Records: {normal_count}")
print(f"Anomaly Percentage: {anomaly_count / len(anomaly_data) * 100:.2f}%")
print(f"Distance Threshold: {threshold:.4f}")

# 7. عرض أول 10 قيم شاذة
anomalies = anomaly_data[
    anomaly_data["anomaly"] == -1
].sort_values(
    by="distance",
    ascending=False
)

print("\nTop 10 Anomalies:")
print(
    anomalies[
        [
            "date",
            "item",
            "price_shekel",
            "usd_rate",
            "border_open",
            "crisis_level",
            "availability",
            "distance"
        ]
    ].head(10).to_string(index=False)
)

# 8. رسم القيم الطبيعية والشاذة
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=anomaly_data,
    x="usd_rate",
    y="price_shekel",
    hue="anomaly",
    palette={1: "blue", -1: "red"},
    alpha=0.7
)

plt.title("Anomaly Detection: Market Prices")
plt.xlabel("USD Exchange Rate")
plt.ylabel("Price (Shekel)")
plt.legend(
    title="Record Type",
    labels=["Normal", "Anomaly"]
)

plt.tight_layout()
plt.savefig("figures/anomaly_detection.png", dpi=300)
plt.show()

# 9. حفظ النتائج
anomaly_data.to_csv(
    "data/anomaly_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nAnomaly Detection Completed Successfully!")
# ==========================================
# PHASE 3: LINEAR REGRESSION
# التنبؤ بسعر السلعة في اليوم التالي
# ==========================================

print("\n" + "=" * 60)
print("PHASE 3: LINEAR REGRESSION")
print("=" * 60)

# 1. ترتيب البيانات حسب السلعة والتاريخ
regression_data = df.sort_values(
    by=["item", "date"]
).copy()

# 2. إنشاء السعر المستهدف (سعر اليوم التالي)
regression_data["price_next"] = (
    regression_data.groupby("item")["price_shekel"].shift(-1)
)

# حذف الصفوف التي لا تملك سعر اليوم التالي
regression_data = regression_data.dropna(
    subset=["price_next"]
).copy()

# 3. تجهيز المتغيرات
features = [
    "price_shekel",
    "usd_rate",
    "border_open",
    "crisis_level",
    "avail_num"
]

# إنشاء avail_num إذا لم يكن موجودًا في df
availability_mapping = {
    "متوفر": 2,
    "ناقص": 1,
    "شحيح": 0
}

regression_data["avail_num"] = (
    regression_data["availability"].map(availability_mapping)
)

X = regression_data[features]
y = regression_data["price_next"]

# 4. التقسيم الزمني 80% تدريب و20% اختبار
unique_dates = sorted(regression_data["date"].unique())

split_position = int(len(unique_dates) * 0.8)
split_date = unique_dates[split_position]

train_mask = regression_data["date"] < split_date
test_mask = regression_data["date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("\nDataset Splitting:")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"Split date: {split_date}")

# 5. توحيد المقاييس
scaler_regression = StandardScaler()

X_train_scaled = scaler_regression.fit_transform(X_train)
X_test_scaled = scaler_regression.transform(X_test)

# 6. تدريب نموذج Linear Regression
linear_model = LinearRegression()

linear_model.fit(X_train_scaled, y_train)

# 7. التنبؤ
y_pred = linear_model.predict(X_test_scaled)

# 8. تقييم النموذج
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

mape = np.mean(
    np.abs((y_test - y_pred) / y_test)
) * 100

print("\nModel Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# 9. عرض بعض التوقعات
prediction_results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(prediction_results.head(15).round(2))

# 10. معاملات النموذج
coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": linear_model.coef_
})

print("\nModel Coefficients:")
print(coefficients.round(4))

# 11. رسم الأسعار الحقيقية والمتوقعة
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Perfect Prediction"
)

plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/linear_regression_results.png",
    dpi=300
)

plt.show()

# 12. حفظ النتائج
prediction_results.to_csv(
    "data/linear_regression_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

coefficients.to_csv(
    "data/linear_regression_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nLinear Regression Completed Successfully!")

# ============================================================
# PHASE 4: LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 4: LOGISTIC REGRESSION")
print("=" * 60)

# 1. Select input features
features = [
    "price_shekel",
    "usd_rate",
    "border_open",
    "crisis_level"
]

X = df[features]
y = df["availability"]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset Splitting:")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 3. Feature scaling
scaler_log = StandardScaler()

X_train_scaled = scaler_log.fit_transform(X_train)
X_test_scaled = scaler_log.transform(X_test)

# 4. Create Logistic Regression model
log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# 5. Train the model
log_model.fit(X_train_scaled, y_train)

# 6. Make predictions
y_pred = log_model.predict(X_test_scaled)

# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")

# 8. Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["متوفر", "ناقص", "شحيح"],
        zero_division=0
    )
)

# 9. Confusion Matrix
class_names = ["متوفر", "ناقص", "شحيح"]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_names
)

print("\nConfusion Matrix:")
print(pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
))

# 10. Save confusion matrix plot
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Availability")
plt.ylabel("Actual Availability")
plt.tight_layout()

plt.savefig(
    "figures/logistic_regression_confusion_matrix.png"
)

plt.close()

# 11. Feature coefficients
coefficients = pd.DataFrame(
    log_model.coef_,
    columns=features,
    index=log_model.classes_
)

print("\nModel Coefficients:")
print(coefficients.round(4))

# Save coefficients
coefficients.to_csv(
    "data/logistic_regression_coefficients.csv"
)

# 12. Save prediction results
prediction_results = pd.DataFrame({
    "Actual Availability": y_test.values,
    "Predicted Availability": y_pred
})

prediction_results.to_csv(
    "data/logistic_regression_predictions.csv",
    index=False
)

print("\nLogistic Regression Completed Successfully!")
print("Results saved successfully!")

# ============================================================
# PHASE 5: MARKET SIMULATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 5: MARKET SIMULATION")
print("=" * 60)

# Convert availability into numeric values
availability_mapping = {
    "متوفر": 2,
    "ناقص": 1,
    "شحيح": 0
}

# Function 1: Predict next price
def predict_price_next(price_today, usd, border, crisis, avail):

    avail_num = availability_mapping[avail]

    input_data = pd.DataFrame([{
        "price_shekel": price_today,
        "usd_rate": usd,
        "border_open": border,
        "crisis_level": crisis,
        "avail_num": avail_num
    }])

    input_scaled = scaler_regression.transform(input_data)

    predicted_price = linear_model.predict(input_scaled)[0]
    return predicted_price


# Function 2: Predict availability
def predict_availability(price_today, usd, border, crisis):

    input_data = pd.DataFrame([{
        "price_shekel": price_today,
        "usd_rate": usd,
        "border_open": border,
        "crisis_level": crisis
    }])

    input_scaled = scaler_log.transform(input_data)

    predicted_availability = log_model.predict(input_scaled)[0]

    return predicted_availability


# 3. Define market scenarios
scenarios = [
    {
        "scenario": "Market Stability",
        "price": 120,
        "usd": 3.70,
        "border": 1,
        "crisis": 0,
        "availability": "متوفر"
    },
    {
        "scenario": "Moderate Escalation",
        "price": 150,
        "usd": 3.85,
        "border": 1,
        "crisis": 1,
        "availability": "ناقص"
    },
    {
        "scenario": "Severe Crisis and Closure",
        "price": 180,
        "usd": 3.95,
        "border": 0,
        "crisis": 2,
        "availability": "شحيح"
    }
]


# 4. Run simulation
simulation_results = []

for scenario in scenarios:

    predicted_price = predict_price_next(
        scenario["price"],
        scenario["usd"],
        scenario["border"],
        scenario["crisis"],
        scenario["availability"]
    )

    predicted_availability = predict_availability(
        scenario["price"],
        scenario["usd"],
        scenario["border"],
        scenario["crisis"]
    )

    simulation_results.append({
        "Scenario": scenario["scenario"],
        "Current Price": scenario["price"],
        "Predicted Next Price": round(predicted_price, 2),
        "Expected Availability": scenario["availability"],
        "Predicted Availability": predicted_availability
    })


# 5. Display results
simulation_df = pd.DataFrame(simulation_results)

print("\nSimulation Results:")
print(simulation_df.to_string(index=False))


# 6. Save simulation results
simulation_df.to_csv(
    "data/market_simulation_results.csv",
    index=False
)

# 7. Plot predicted prices
plt.figure(figsize=(10, 6))

plt.bar(
    simulation_df["Scenario"],
    simulation_df["Predicted Next Price"]
)

plt.title("Market Simulation - Predicted Prices")
plt.xlabel("Market Scenario")
plt.ylabel("Predicted Price (Shekel)")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("figures/market_simulation.png")
plt.close()

print("\nSimulation Completed Successfully!")
print("Simulation results saved successfully!")
