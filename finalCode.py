#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =========================================================
# FINAL CLEAN VERSION
# Section 1. Data loading and initial screening
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("IBM-HR-Employee-Attrition.csv")

# Basic information
print("Shape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData info:")
df.info()

print("\nFirst 5 rows:")
display(df.head())

# Missing values
print("\nMissing values by column:")
print(df.isnull().sum().sort_values(ascending=False))

# Attrition distribution
print("\nAttrition distribution:")
print(df["Attrition"].value_counts())

plt.figure(figsize=(6, 4))
df["Attrition"].value_counts().plot(kind="bar")
plt.title("Attrition Distribution")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# JobRole distribution
print("\nJobRole counts:")
print(df["JobRole"].value_counts())

plt.figure(figsize=(10, 5))
df["JobRole"].value_counts().plot(kind="bar")
plt.title("JobRole Distribution")
plt.xlabel("JobRole")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.show()

# Salary summary by JobRole
salary_stats = df.groupby("JobRole")["MonthlyIncome"].agg(["mean", "median", "min", "max", "count"])
print("\nMonthlyIncome summary by JobRole:")
display(salary_stats)

# Drop invalid / non-informative columns
drop_cols = [
    "EmployeeNumber",   # ID variable
    "EmployeeCount",    # constant column
    "Over18",           # constant column
    "StandardHours"     # constant column
]

df = df.drop(columns=drop_cols)

print("\nShape after dropping unused columns:", df.shape)
print("\nRemaining columns:")
print(df.columns.tolist())


# In[2]:


# =========================================================
# Section 1B. Exploratory check for JobLevel vs experience
# =========================================================

joblevel_exp = df.groupby("JobLevel")["TotalWorkingYears"].agg(
    ["count", "mean", "median", "min", "max"]
)
print("TotalWorkingYears by JobLevel:")
display(joblevel_exp)

jobrole_joblevel_exp = (
    df.groupby(["JobRole", "JobLevel"])["TotalWorkingYears"]
    .agg(["count", "mean", "median", "min", "max"])
    .reset_index()
)

print("\nTotalWorkingYears by JobRole and JobLevel:")
display(jobrole_joblevel_exp)


# In[3]:


# =========================================================
# Section 1C. Exploratory Data Analysis (Pre-construction)
# =========================================================

import seaborn as sns

# This section provides descriptive exploration of key raw variables
# before treatment construction.

# ------------------------------------------
# 1. Attrition vs MonthlyIncome
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income by Attrition")
plt.show()

# ------------------------------------------
# 2. JobLevel vs Attrition
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="JobLevel", hue="Attrition", data=df)
plt.title("Attrition by JobLevel")
plt.show()

# ------------------------------------------
# 3. JobRole vs Attrition
# ------------------------------------------
plt.figure(figsize=(10, 5))
sns.countplot(x="JobRole", hue="Attrition", data=df)
plt.xticks(rotation=45, ha="right")
plt.title("Attrition by JobRole")
plt.show()

# ------------------------------------------
# 4. OverTime vs Attrition
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="OverTime", hue="Attrition", data=df)
plt.title("Attrition by OverTime")
plt.show()

# ------------------------------------------
# 5. Satisfaction variables vs Attrition
# ------------------------------------------
satisfaction_vars = [
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance"
]

for var in satisfaction_vars:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Attrition", y=var, data=df)
    plt.title(f"{var} by Attrition")
    plt.show()


# In[4]:


# =========================================================
# Section 2. Variable construction
# =========================================================

# Outcome variable: Attrition -> binary
df["Attrition_binary"] = df["Attrition"].map({"Yes": 1, "No": 0})

print("Attrition_binary distribution:")
print(df["Attrition_binary"].value_counts())

# Binary variable: OverTime -> 0/1
df["OverTime_binary"] = df["OverTime"].map({"Yes": 1, "No": 0})

print("\nOverTime_binary distribution:")
print(df["OverTime_binary"].value_counts())

# External market salary benchmark using Glassdoor base pay midpoint (annual USD)
market_salary_dict = {
    "Healthcare Representative": 67500,   # (51k + 84k)/2
    "Human Resources": 84500,             # (60k + 109k)/2
    "Laboratory Technician": 55000,       # (45k + 65k)/2
    "Manager": 83000,                     # (60k + 106k)/2
    "Manufacturing Director": 121500,     # (91k + 152k)/2
    "Research Director": 159500,          # (116k + 203k)/2
    "Research Scientist": 154500,         # (122k + 187k)/2
    "Sales Executive": 90000,             # (63k + 117k)/2
    "Sales Representative": 78500         # (60k + 97k)/2
}

# Map benchmark to dataset
df["MarketSalary_external_annual"] = df["JobRole"].map(market_salary_dict)
df["MarketSalary_external_monthly"] = df["MarketSalary_external_annual"] / 12

# Check whether any JobRole failed to match
missing_benchmark = df["MarketSalary_external_monthly"].isnull().sum()
print(f"\nNumber of missing external benchmark values: {missing_benchmark}")

print("\nExternal market benchmark preview:")
display(
    df[["JobRole", "MonthlyIncome", "MarketSalary_external_annual", "MarketSalary_external_monthly"]]
    .drop_duplicates(subset=["JobRole"])
    .sort_values("JobRole")
    .reset_index(drop=True)
)

print("\nExternal market benchmark summary (monthly):")
print(df["MarketSalary_external_monthly"].describe())


# In[5]:


# =========================================================
# Section 3. Treatment benchmark matching
# =========================================================

# Construct treatment variable: salary-market alignment ratio
df["SalaryRatio"] = df["MonthlyIncome"] / df["MarketSalary_external_monthly"]

print("SalaryRatio summary:")
print(df["SalaryRatio"].describe())

print("\nSalaryRatio percentiles:")
print(df["SalaryRatio"].quantile([0.01, 0.05, 0.50, 0.95, 0.99]))

# Winsorize the treatment variable
lower = df["SalaryRatio"].quantile(0.01)
upper = df["SalaryRatio"].quantile(0.99)

df["SalaryRatio_winsorized"] = df["SalaryRatio"].clip(lower=lower, upper=upper)

print("\nSalaryRatio_winsorized summary:")
print(df["SalaryRatio_winsorized"].describe())

# Plot treatment distribution
plt.figure(figsize=(8, 5))
plt.hist(df["SalaryRatio_winsorized"], bins=30)
plt.title("Distribution of SalaryRatio_winsorized")
plt.xlabel("Salary Ratio (Winsorized)")
plt.ylabel("Frequency")
plt.show()


# In[6]:


# =========================================================
# Section 3B. Exploratory analysis of SalaryRatio
# =========================================================

# ------------------------------------------
# 1. SalaryRatio vs Attrition
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(x="Attrition", y="SalaryRatio_winsorized", data=df)
plt.title("Salary Ratio by Attrition")
plt.show()

# ------------------------------------------
# 2. SalaryRatio distribution by Attrition
# ------------------------------------------
plt.figure(figsize=(6, 4))
sns.kdeplot(data=df, x="SalaryRatio_winsorized", hue="Attrition", fill=True)
plt.title("Salary Ratio Distribution by Attrition")
plt.show()

# ------------------------------------------
# 3. SalaryRatio overall distribution
# ------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["SalaryRatio_winsorized"], bins=30)
plt.title("Distribution of Salary Ratio")
plt.xlabel("Salary Ratio")
plt.ylabel("Frequency")
plt.show()

# ------------------------------------------
# 4. Correlation matrix (selected variables)
# ------------------------------------------
selected_vars = [
    "Attrition_binary",
    "SalaryRatio_winsorized",
    "JobLevel",
    "OverTime_binary",
    "JobSatisfaction",
    "WorkLifeBalance",
    "TotalWorkingYears",
    "YearsAtCompany"
]

corr = df[selected_vars].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix (Selected Variables)")
plt.show()


# In[7]:


# =========================================================
# Section 4. Data preprocessing (UPDATED - include JobRole)
# =========================================================

from sklearn.preprocessing import StandardScaler

# Quick check of key variables
print(df[["Attrition_binary", "OverTime_binary", "SalaryRatio_winsorized"]].head())
print("\nMissing values in key variables:")
print(df[["Attrition_binary", "OverTime_binary", "SalaryRatio_winsorized"]].isnull().sum())

# Covariates (NOW INCLUDING JobRole)
X_cols = [
    "Age",
    "JobLevel",
    "YearsAtCompany",
    "OverTime_binary",
    "StockOptionLevel",
    "PerformanceRating",
    "TotalWorkingYears",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "JobRole"   # ⭐ NEW (IMPORTANT)
]

# Define outcome and treatment
Y = df["Attrition_binary"].values
T = df["SalaryRatio_winsorized"].values

# Build X
X = df[X_cols].copy()

# 🔥 One-hot encode JobRole
X = pd.get_dummies(X, columns=["JobRole"], drop_first=True, dtype=int)

print("\nX after adding JobRole dummies:")
print(X.shape)
display(X.head())

# Standardize continuous variables ONLY
continuous_cols = [
    "Age",
    "YearsAtCompany",
    "TotalWorkingYears",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_cols] = scaler.fit_transform(X_scaled[continuous_cols])

print("\nScaled covariates preview:")
display(X_scaled.head())


# In[8]:


# =========================================================
# Section 5. Main causal estimation: CausalForestDML
# =========================================================

from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

# JobLevel is included in X to account for internal hierarchy differences

# Outcome model
model_y = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

# Treatment model
model_t = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

# Causal forest model
cf = CausalForestDML(
    model_y=model_y,
    model_t=model_t,
    n_estimators=500,
    min_samples_leaf=10,
    max_depth=10,
    discrete_treatment=False,
    cv=3,
    random_state=42,
    n_jobs=-1
)

# Fit model
cf.fit(Y, T, X=X_scaled)

print("CausalForestDML model fitted successfully.")
print("Treatment variable: SalaryRatio_winsorized")
print("Key control variables included: JobLevel and JobRole (as confounders)")

# Estimated individual treatment effects (CATE)
cate = cf.effect(X_scaled)
df["CATE"] = cate

print("\nCATE summary:")
print(df["CATE"].describe())

# Average estimated treatment effect (ATE)
ate = np.mean(cate)
print("\nAverage estimated treatment effect:", ate)

# More interpretable scale: 10% increase in salary ratio
ate_10pct = ate * 0.1
print("Effect of 10% increase in salary ratio:", ate_10pct)

# Confidence interval
try:
    ate_interval = cf.ate_interval(X=X_scaled)
    print("ATE 95% confidence interval:", ate_interval)
except Exception as e:
    print("ATE interval not available in current setup:", e)

# Plot CATE distribution
plt.figure(figsize=(8, 5))
plt.hist(df["CATE"], bins=30)
plt.title("Distribution of Estimated CATE")
plt.xlabel("Estimated Treatment Effect")
plt.ylabel("Frequency")
plt.show()


# In[9]:


# =========================================================
# Section 6. Heterogeneous treatment effect analysis
# =========================================================

from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 1. Heterogeneity by JobRole
# ------------------------
jobrole_cate = (
    df.groupby("JobRole")["CATE"]
    .agg(["mean", "std", "count"])
)

# Add t-test p-values
p_values = []
for role in jobrole_cate.index:
    vals = df.loc[df["JobRole"] == role, "CATE"]
    t_stat, p_val = ttest_1samp(vals, popmean=0)
    p_values.append(p_val)

jobrole_cate["p_value"] = p_values
jobrole_cate = jobrole_cate.sort_values("mean")

print("CATE summary by JobRole (with significance):")
display(jobrole_cate)

# ------------------------
# 2. JobRole clusters
# ------------------------
jobrole_cluster_map = {
    "Sales Executive": "Sales",
    "Sales Representative": "Sales",
    "Research Scientist": "Technical",
    "Laboratory Technician": "Technical",
    "Research Director": "Technical",
    "Manufacturing Director": "Managerial",
    "Manager": "Managerial",
    "Healthcare Representative": "Support",
    "Human Resources": "Support"
}

df["JobRoleCluster"] = df["JobRole"].map(jobrole_cluster_map)

cluster_cate = (
    df.groupby("JobRoleCluster")["CATE"]
    .agg(["mean", "std", "count"])
)

# Add p-values
p_values = []
for group in cluster_cate.index:
    vals = df.loc[df["JobRoleCluster"] == group, "CATE"]
    t_stat, p_val = ttest_1samp(vals, popmean=0)
    p_values.append(p_val)

cluster_cate["p_value"] = p_values
cluster_cate = cluster_cate.sort_values("mean")

print("\nCATE summary by JobRoleCluster (with significance):")
display(cluster_cate)

# ------------------------
# 3. Market pressure groups
# ------------------------

df["MarketPressureGroup"] = np.where(
    df["SalaryRatio_winsorized"] < 1,
    "High Pressure",
    "Low Pressure"
)

pressure_cate = (
    df.groupby("MarketPressureGroup")["CATE"]
    .agg(["mean", "std", "count"])
)

print(df["MarketPressureGroup"].value_counts())

# Add p-values
p_values = []
for group in pressure_cate.index:
    vals = df.loc[df["MarketPressureGroup"] == group, "CATE"]
    t_stat, p_val = ttest_1samp(vals, popmean=0)
    p_values.append(p_val)

pressure_cate["p_value"] = p_values
pressure_cate = pressure_cate.sort_values("mean")

print("\nCATE summary by MarketPressureGroup (with significance):")
display(pressure_cate)

# ------------------------
# 4. Heterogeneity by JobLevel
# ------------------------
joblevel_cate = (
    df.groupby("JobLevel")["CATE"]
    .agg(["mean", "std", "count"])
)

# Add p-values
p_values = []
for level in joblevel_cate.index:
    vals = df.loc[df["JobLevel"] == level, "CATE"]
    t_stat, p_val = ttest_1samp(vals, popmean=0)
    p_values.append(p_val)

joblevel_cate["p_value"] = p_values
joblevel_cate = joblevel_cate.sort_values("mean")

print("\nCATE summary by JobLevel (with significance):")
display(joblevel_cate)

# ------------------------
# 5. Visualization
# ------------------------

# Plot by JobLevel
plt.figure(figsize=(7, 4))
joblevel_cate["mean"].plot(kind="bar")
plt.title("Average Treatment Effect by JobLevel")
plt.ylabel("Mean CATE")
plt.xlabel("JobLevel")
plt.axhline(0, linestyle="--")
plt.xticks(rotation=0)
plt.show()

# Plot by JobRoleCluster
plt.figure(figsize=(8, 5))
cluster_cate["mean"].plot(kind="bar")
plt.title("Average Treatment Effect by JobRoleCluster")
plt.ylabel("Mean CATE")
plt.xlabel("JobRoleCluster")
plt.axhline(0, linestyle="--")
plt.xticks(rotation=0)
plt.show()

# Plot by MarketPressureGroup
plt.figure(figsize=(6, 4))
pressure_cate["mean"].plot(kind="bar")
plt.title("Average Treatment Effect by Market Pressure")
plt.ylabel("Mean CATE")
plt.xlabel("MarketPressureGroup")
plt.axhline(0, linestyle="--")
plt.xticks(rotation=0)
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# Keep the order clear
order = ["High Pressure", "Low Pressure"]
plot_data = pressure_cate.loc[order, "mean"]

fig, ax = plt.subplots(figsize=(6, 4))

x = np.arange(len(plot_data))

# Bar plot
bars = ax.bar(x, plot_data.values, edgecolor="black", linewidth=1)

# Add zero line
ax.axhline(0, linestyle="--", linewidth=1)

# Add marker points to make very small values visible
ax.scatter(x, plot_data.values, s=60, color="black", zorder=3)

# Add value labels
for i, value in enumerate(plot_data.values):
    if value < -0.001:
        ax.text(i, value - 0.002, f"{value:.5f}", ha="center", va="top")
    else:
        ax.text(i, value - 0.003, f"{value:.5f}", ha="center", va="top")

ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_title("Average Treatment Effect by Market Pressure")
ax.set_ylabel("Mean CATE")
ax.set_xlabel("Market Pressure Group")

ax.set_ylim(-0.04, 0.005)

plt.tight_layout()
plt.show()


# In[10]:


# ------------------------
# 6. Interaction: JobRole × JobLevel (NEW)
# ------------------------

interaction_cate = (
    df.groupby(["JobRole", "JobLevel"])["CATE"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

# filter away small samples
interaction_cate = interaction_cate[interaction_cate["count"] >= 20]

print("\nCATE by JobRole × JobLevel (interaction):")
display(interaction_cate.sort_values("mean"))


# In[11]:


# ------------------------
# 7. Interaction: JobRoleCluster × JobLevel
# ------------------------

interaction_cluster_cate = (
    df.groupby(["JobRoleCluster", "JobLevel"])["CATE"]
    .agg(["mean", "std", "count"])
    .reset_index()
)

interaction_cluster_cate = interaction_cluster_cate[interaction_cluster_cate["count"] >= 20]

print("\nCATE by JobRoleCluster × JobLevel:")
display(interaction_cluster_cate.sort_values("mean"))

# ------------------------
# Visualization: Cluster × JobLevel
# ------------------------

import seaborn as sns

plt.figure(figsize=(8, 5))

sns.barplot(
    data=interaction_cluster_cate,
    x="JobLevel",
    y="mean",
    hue="JobRoleCluster"
)

plt.title("Interaction Effect: JobRoleCluster × JobLevel")
plt.ylabel("Mean CATE")
plt.xlabel("JobLevel")
plt.axhline(0, linestyle="--")
plt.legend(title="Cluster")
plt.show()


# In[12]:


# ------------------------
# Heatmap 
# ------------------------

pivot_table = interaction_cluster_cate.pivot(
    index="JobRoleCluster",
    columns="JobLevel",
    values="mean"
)

plt.figure(figsize=(7, 4))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm", center=0)

plt.title("CATE Heatmap: Cluster × JobLevel")
plt.show()


# In[13]:


# =========================================================
# Section 7. Statistical summary tests
# (Subgroup tests only; full-sample t-test removed)
# =========================================================

import pandas as pd
from scipy.stats import ttest_1samp

# ------------------------
# 1. JobRoleCluster tests
# ------------------------
cluster_results = []

for group in sorted(df["JobRoleCluster"].dropna().unique()):
    vals = df.loc[df["JobRoleCluster"] == group, "CATE"]
    t_stat, p_value = ttest_1samp(vals, popmean=0)
    
    cluster_results.append({
        "Group": group,
        "Mean CATE": vals.mean(),
        "Std": vals.std(),
        "N": len(vals),
        "t-stat": t_stat,
        "p-value": p_value
    })

cluster_results_df = pd.DataFrame(cluster_results).sort_values("Mean CATE")

print("JobRoleCluster results:")
display(cluster_results_df)

# ------------------------
# 2. MarketPressureGroup tests
# ------------------------
pressure_results = []

for group in sorted(df["MarketPressureGroup"].dropna().unique()):
    vals = df.loc[df["MarketPressureGroup"] == group, "CATE"]
    t_stat, p_value = ttest_1samp(vals, popmean=0)
    
    pressure_results.append({
        "Group": group,
        "Mean CATE": vals.mean(),
        "Std": vals.std(),
        "N": len(vals),
        "t-stat": t_stat,
        "p-value": p_value
    })

pressure_results_df = pd.DataFrame(pressure_results).sort_values("Mean CATE")

print("\nMarketPressureGroup results:")
display(pressure_results_df)

# ------------------------
# 3. JobLevel tests
# ------------------------
joblevel_results = []

for level in sorted(df["JobLevel"].dropna().unique()):
    vals = df.loc[df["JobLevel"] == level, "CATE"]
    t_stat, p_value = ttest_1samp(vals, popmean=0)
    
    joblevel_results.append({
        "Group": f"Level {level}",
        "Mean CATE": vals.mean(),
        "Std": vals.std(),
        "N": len(vals),
        "t-stat": t_stat,
        "p-value": p_value
    })

joblevel_results_df = pd.DataFrame(joblevel_results).sort_values("Mean CATE")

print("\nJobLevel results:")
display(joblevel_results_df)


# In[14]:


# ------------------------
# 4. Interaction tests: JobRoleCluster × JobLevel
# ------------------------

interaction_results = []

for cluster in df["JobRoleCluster"].dropna().unique():
    for level in df["JobLevel"].dropna().unique():
        
        subset = df[
            (df["JobRoleCluster"] == cluster) &
            (df["JobLevel"] == level)
        ]["CATE"]
        
        # 避免样本太小
        if len(subset) >= 20:
            t_stat, p_value = ttest_1samp(subset, popmean=0)
            
            interaction_results.append({
                "Cluster": cluster,
                "JobLevel": level,
                "Mean CATE": subset.mean(),
                "Std": subset.std(),
                "N": len(subset),
                "t-stat": t_stat,
                "p-value": p_value
            })

interaction_results_df = pd.DataFrame(interaction_results)

print("\nInteraction results (Cluster × JobLevel):")
display(interaction_results_df.sort_values("Mean CATE"))


# In[15]:


# =========================================================
# Section 8. SHAP-based validation
# =========================================================

import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


# This section provides predictive validation only.
# SHAP values reflect predictive associations, not causal effects.

# ------------------------------------------
# 1. Build predictive dataset
# ------------------------------------------
predictor_features = X_scaled.copy()
predictor_features["SalaryRatio_winsorized"] = df["SalaryRatio_winsorized"].values

print("Predictor feature shape:", predictor_features.shape)
print("Outcome shape:", Y.shape)

# ------------------------------------------
# 2. Train/test split
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    predictor_features,
    Y,
    test_size=0.3,
    random_state=42,
    stratify=Y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ------------------------------------------
# 3. Fit predictive model
# ------------------------------------------
rf_clf = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_clf.fit(X_train, y_train)

print("\nRandomForestClassifier fitted successfully.")

# ------------------------------------------
# 4. Predictive performance check
# ------------------------------------------
train_prob = rf_clf.predict_proba(X_train)[:, 1]
test_prob = rf_clf.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_prob)
test_auc = roc_auc_score(y_test, test_prob)

print("Train AUC:", train_auc)
print("Test AUC:", test_auc)

# Optional classification report on test set
y_test_pred = rf_clf.predict(X_test)
print("\nClassification report on test set:")
print(classification_report(y_test, y_test_pred))

# ------------------------------------------
# 5. SHAP explanation
#    Use training data for SHAP interpretation
# ------------------------------------------
explainer = shap.TreeExplainer(rf_clf)
shap_values = explainer.shap_values(X_train)

print("\nRaw shap_values shape:", np.array(shap_values).shape)
print("X_train shape:", X_train.shape)

# For your SHAP version: (n_samples, n_features, n_classes)
shap_matrix = shap_values[:, :, 1]

print("Final shap_matrix shape:", shap_matrix.shape)

# ------------------------------------------
# 6. SHAP summary plot
# ------------------------------------------
print("\nGenerating SHAP summary plot...")
shap.summary_plot(shap_matrix, X_train)

# ------------------------------------------
# 7. SHAP dependence plots
# ------------------------------------------
    
for feature_name in [
    "SalaryRatio_winsorized",
    "OverTime_binary",
    "JobSatisfaction",
    "JobRole_Manager"
]:
    print(f"\nGenerating SHAP dependence plot for: {feature_name}")
    shap.dependence_plot(feature_name, shap_matrix, X_train)


# In[16]:


shap.dependence_plot(
    "SalaryRatio_winsorized",
    shap_matrix,
    X_train,
    interaction_index=None
)


# In[17]:


import matplotlib as mpl
from textwrap import fill

# Set font to Times New Roman
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 10

# -----------------------------
# Appendix A variable table data
# -----------------------------

data = [
    ("Age", "Employee age"),
    ("Attrition", "Employee left the company or stayed"),
    ("BusinessTravel", "Frequency of business travel"),
    ("DailyRate", "Daily rate value of the employee"),
    ("Department", "Department of the employee"),
    ("DistanceFromHome", "Distance between the employee’s home and workplace"),
    ("Education", "Education level of the employee"),
    ("EducationField", "Field of education of the employee"),
    ("EmployeeCount", "Employee count value in the dataset"),
    ("EmployeeNumber", "Unique employee identification number"),
    ("EnvironmentSatisfaction", "Employee satisfaction with the work environment"),
    ("Gender", "Gender of the employee"),
    ("HourlyRate", "Hourly rate value of the employee"),
    ("JobInvolvement", "Level of employee job involvement"),
    ("JobLevel", "Job level of the employee within the organization"),
    ("JobRole", "Job role of the employee"),
    ("JobSatisfaction", "Employee satisfaction with the job"),
    ("MaritalStatus", "Marital status of the employee"),
    ("MonthlyIncome", "Monthly income of the employee"),
    ("MonthlyRate", "Monthly rate value of the employee"),
    ("NumCompaniesWorked", "Number of companies the employee has worked for before"),
    ("Over18", "Whether the employee is over 18 years old"),
    ("OverTime", "Whether the employee works overtime"),
    ("PercentSalaryHike", "Percentage increase in salary"),
    ("PerformanceRating", "Performance rating of the employee"),
    ("RelationshipSatisfaction", "Employee satisfaction with workplace relationships"),
    ("StandardHours", "Standard working hours"),
    ("StockOptionLevel", "Stock option level of the employee"),
    ("TotalWorkingYears", "Total number of working years of the employee"),
    ("TrainingTimesLastYear", "Number of training times attended by the employee last year"),
    ("WorkLifeBalance", "Employee rating of work-life balance"),
    ("YearsAtCompany", "Number of years the employee has worked at the company"),
    ("YearsInCurrentRole", "Number of years the employee has worked in the current role"),
    ("YearsSinceLastPromotion", "Number of years since the employee’s last promotion"),
    ("YearsWithCurrManager", "Number of years the employee has worked with the current manager")
]

df = pd.DataFrame(data, columns=["Column Name", "Description"])

# Wrap long text so it fits inside the image
df["Description"] = df["Description"].apply(lambda x: fill(x, width=65))

# -----------------------------
# Create table image
# -----------------------------

fig_height = max(14, len(df) * 0.45)
fig, ax = plt.subplots(figsize=(12, fig_height))
ax.axis("off")

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    cellLoc="left",
    colLoc="left",
    colWidths=[0.25, 0.75],
    loc="center"
)

# Basic style
table.auto_set_font_size(False)
table.set_fontsize(10)

# Style header and cells
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("black")
    cell.set_linewidth(0.8)

    if row == 0:
        cell.set_text_props(
            weight="bold",
            fontfamily="Times New Roman",
            fontsize=10
        )
        cell.set_height(0.04)
    else:
        cell.set_text_props(
            fontfamily="Times New Roman",
            fontsize=10
        )
        cell.set_height(0.035)

# Save image
plt.savefig(
    "Appendix_A_variable_description.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Saved as Appendix_A_variable_description.png")


# In[ ]:




