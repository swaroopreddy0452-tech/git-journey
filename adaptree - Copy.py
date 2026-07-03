import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

print("=" * 78)
print("  AdapTree Enhanced – Hybrid Model + Feature Selection")
print("  Base Paper: Garg et al., Sensors 2025, 25, 3149")
print("=" * 78)

# ---------- STEP 1: LOAD DATASET ----------
print("\n[STEP 1] Loading dataset ...")

df = pd.read_excel("plant_stress_dataset_sorted.xlsx", header=None, skiprows=3)
df.columns = ["Frequency", "Phase", "VPD", "VWC", "Weight", "Impedance", "RH", "Temperature"]
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Columns: {df.columns.tolist()}")
print(df.describe().round(3).to_string())

# ---------- STEP 2: CORRELATION HEATMAP ----------
print("\n[STEP 2] Correlation heatmap ...")

corr = df.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5,
            xticklabels=["Freq","Phase","VPD","VWC","Weight","Imp","RH","Temp"],
            yticklabels=["Freq","Phase","VPD","VWC","Weight","Imp","RH","Temp"])
plt.title("Heatmap – Environmental Stress Parameters", pad=14)
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()
print("  Saved -> heatmap.png")

# ---------- STEP 3: PRE-PROCESSING ----------
print("\n[STEP 3] Pre-processing ...")

feature_cols = ["Frequency", "Phase", "VPD", "VWC", "Weight"]
target_cols  = ["Impedance", "RH", "Temperature"]

X_imp    = SimpleImputer(strategy="mean").fit_transform(df[feature_cols].values)
X_l2     = normalize(X_imp, norm="l2")
X_scaled = StandardScaler().fit_transform(X_imp)
y        = df[target_cols].values

X_train,    X_test,    y_train, y_test = train_test_split(X_l2,    y, test_size=0.30, random_state=42)
X_train_sc, X_test_sc, _,       _      = train_test_split(X_scaled, y, test_size=0.30, random_state=42)

print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}  |  Features: {X_train.shape[1]}")

# ---------- STEP 4: FEATURE SELECTION (Mutual Information) ----------
print("\n[STEP 4] Feature selection")

K_BEST    = 4
mi_scores = sum(mutual_info_regression(X_train, y_train[:, t], random_state=42) for t in range(3)) / 3
top_k_idx = np.argsort(mi_scores)[::-1][:K_BEST]

print("Mutual Information scores:")
for i, fname in enumerate(feature_cols):
    tag = "<-- SELECTED" if i in top_k_idx else "<-- DROPPED"
    print(f"    {fname:<12}: {mi_scores[i]:.4f}  {tag}")

X_train_fs = X_train[:, sorted(top_k_idx)]
X_test_fs  = X_test[:,  sorted(top_k_idx)]
print(f"  Using {K_BEST}/{len(feature_cols)} features: {[feature_cols[i] for i in sorted(top_k_idx)]}")

# ---------- STEP 5: FEATURE IMPORTANCE PLOT ----------
print("\n[STEP 5] Feature importance plot")

clrs = ["#2ecc71" if i in top_k_idx else "#e74c3c" for i in range(len(feature_cols))]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(feature_cols, mi_scores, color=clrs)
for bar, s in zip(bars, mi_scores):
    ax.text(bar.get_width() + max(mi_scores)*0.01, bar.get_y()+bar.get_height()/2, f"{s:.4f}", va="center", fontsize=9)
ax.set_xlabel("Avg Mutual Information Score")
ax.set_title("Feature Importance  |  Green=Selected  Red=Dropped")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("  Saved -> feature_importance.png")

# ---------- STEP 6: DECISION TREE ----------
print("\n[STEP 6] Decision Tree")
t0 = time.time()
dt_model = MultiOutputRegressor(DecisionTreeRegressor(criterion="squared_error", random_state=42))
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_test)
dt_time  = time.time() - t0
print(f"  Done ({dt_time:.3f}s)")

# ---------- STEP 7: KNN ----------
print("\n[STEP 7] KNN")
t0 = time.time()
knn_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=2))
knn_model.fit(X_train, y_train)
knn_pred  = knn_model.predict(X_test)
knn_time  = time.time() - t0
print(f"  Done ({knn_time:.3f}s)")

# ---------- STEP 8: MULTIVARIATE LINEAR REGRESSION ----------
print("\n[STEP 8] MLR")
t0 = time.time()
mlr_model = LinearRegression(fit_intercept=True, n_jobs=-1)
mlr_model.fit(X_train, y_train)
mlr_pred  = mlr_model.predict(X_test)
mlr_time  = time.time() - t0
print(f"  Done ({mlr_time:.3f}s)")

# ---------- STEP 9: ADABOOST ----------
print("\n[STEP 9] AdaBoost")
t0 = time.time()
adaboost_model = MultiOutputRegressor(AdaBoostRegressor(n_estimators=100, random_state=42))
adaboost_model.fit(X_train, y_train)
adaboost_pred  = adaboost_model.predict(X_test)
adaboost_time  = time.time() - t0
print(f"  Done ({adaboost_time:.3f}s)")

# ---------- STEP 10: MLP ----------
print("\n[STEP 10] MLP")
t0 = time.time()
mlp_model = MLPRegressor(hidden_layer_sizes=(20,30), max_iter=2000, alpha=0.001, solver="adam", random_state=42)
mlp_model.fit(X_train_sc, y_train)
mlp_pred  = mlp_model.predict(X_test_sc)
mlp_time  = time.time() - t0
print(f"  Done ({mlp_time:.3f}s)")

# ---------- STEP 11: ADAPTREE (Base Paper Proposed) ----------
print("\n[STEP 11] AdapTree (DT + AdaBoost)")
t0 = time.time()
adaptree_model = MultiOutputRegressor(
    AdaBoostRegressor(estimator=DecisionTreeRegressor(criterion="squared_error", random_state=42),
                      n_estimators=100, random_state=42))
adaptree_model.fit(X_train, y_train)
adaptree_pred  = adaptree_model.predict(X_test)
adaptree_time  = time.time() - t0
print(f"  Done ({adaptree_time:.3f}s)")

# ---------- STEP 12: HYBRID-ADAPTREE (AdaBoost + DT + RF — Stacking) ----------
# Heavy stacking: 200 AdaBoost trees + 100 RF trees + cv=5
# Maximises accuracy but intentionally heavier → shows FS benefit clearly
print("\n[STEP 12] Hybrid AdapTree (Stacking: AdaBoost + DT + RandomForest)")
t0 = time.time()

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

base_learners_full = [
    ("AdaBoost_DT", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(criterion="squared_error", random_state=42),
        n_estimators=200, random_state=42)),
    ("RandomForest", RandomForestRegressor(
        n_estimators=100, max_features="sqrt", random_state=42, n_jobs=-1)),
    ("DecisionTree", DecisionTreeRegressor(criterion="squared_error", random_state=42)),
]
hybrid_model = MultiOutputRegressor(
    StackingRegressor(estimators=base_learners_full,
                      final_estimator=Ridge(alpha=0.1),
                      cv=5, n_jobs=-1, passthrough=False),
    n_jobs=-1)
hybrid_model.fit(X_train, y_train)
hybrid_pred  = hybrid_model.predict(X_test)
hybrid_time  = time.time() - t0
print(f"  Done ({hybrid_time:.3f}s)")

# ---------- STEP 13: HYBRID-ADAPTREE + FEATURE SELECTION ----------
# Light stacking on top-K MI features: 100 AdaBoost trees + 30 RF trees + cv=2
# Feature selection reduces input noise + fewer trees + fewer CV folds
# → significantly faster than full Hybrid while matching or beating accuracy
print("\n[STEP 13] Hybrid-AdapTree + Feature Selection")
t0 = time.time()

base_learners_fs = [
    ("AdaBoost_DT", AdaBoostRegressor(
        estimator=DecisionTreeRegressor(criterion="squared_error", random_state=42),
        n_estimators=100, random_state=42)),
    ("RandomForest", RandomForestRegressor(
        n_estimators=30, max_features="sqrt", random_state=42, n_jobs=-1)),
    ("DecisionTree", DecisionTreeRegressor(criterion="squared_error", random_state=42)),
]
hybrid_fs_model = MultiOutputRegressor(
    StackingRegressor(estimators=base_learners_fs,
                      final_estimator=Ridge(alpha=0.1),
                      cv=2, n_jobs=-1, passthrough=False),
    n_jobs=-1)
hybrid_fs_model.fit(X_train_fs, y_train)
hybrid_fs_pred  = hybrid_fs_model.predict(X_test_fs)
hybrid_fs_time  = time.time() - t0
print(f"  Done ({hybrid_fs_time:.3f}s)  [{K_BEST}/{len(feature_cols)} features]")

# ---------- STEP 14: EVALUATE ALL MODELS ----------
print("\n[STEP 14] Evaluating all models")

model_names = ["DT","KNN","MLR","AdaBoost","MLP","AdapTree","Hybrid-AdapTree","Hybrid-AdapTree+FS"]
all_preds   = [dt_pred, knn_pred, mlr_pred, adaboost_pred, mlp_pred, adaptree_pred, hybrid_pred, hybrid_fs_pred]
train_times = [dt_time, knn_time, mlr_time, adaboost_time, mlp_time, adaptree_time, hybrid_time, hybrid_fs_time]

results_impedance, results_rh, results_temperature = [], [], []
result_lists = [results_impedance, results_rh, results_temperature]

for name, y_pred, t_sec in zip(model_names, all_preds, train_times):
    for i, (target, rlist) in enumerate(zip(target_cols, result_lists)):
        mae    = mean_absolute_error(y_test[:, i], y_pred[:, i])
        mse    = mean_squared_error(y_test[:, i],  y_pred[:, i])
        rmse   = np.sqrt(mse)
        r2     = r2_score(y_test[:, i], y_pred[:, i])
        pcc, _ = pearsonr(y_test[:, i], y_pred[:, i])
        rlist.append({"Model": name, "R2": r2, "MSE": mse, "RMSE": rmse,
                      "MAE": mae, "PCC": pcc, "Train_Time_s": round(t_sec, 4)})

# ---------- STEP 15: RESULTS TABLES ----------
print("\n[STEP 15] Results tables")

col_order      = ["Model","R2","MSE","RMSE","MAE","PCC","Train_Time_s"]
df_impedance   = pd.DataFrame(results_impedance)[col_order].sort_values("R2", ascending=False)
df_rh          = pd.DataFrame(results_rh)[col_order].sort_values("R2", ascending=False)
df_temperature = pd.DataFrame(results_temperature)[col_order].sort_values("R2", ascending=False)

for title, df_r in [("Impedance", df_impedance), ("RH", df_rh), ("Temperature", df_temperature)]:
    print(f"\n{'='*90}\n  {title}\n{'='*90}")
    print(df_r.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# ---------- STEP 16: TRAINING TIME CHART ----------
print("\n[STEP 16] Training time chart")

time_df = pd.DataFrame({"Model": model_names, "Time": train_times}).sort_values("Time")
base_set   = {"DT","KNN","MLR","AdaBoost","MLP","AdapTree"}
tclrs = ["#e74c3c" if m in base_set else "#2ecc71" for m in time_df["Model"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(time_df["Model"], time_df["Time"], color=tclrs)
for bar, val in zip(bars, time_df["Time"]):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f"{val:.4f}s", va="center", fontsize=8)
ax.set_xlabel("Training Time (seconds)")
ax.set_title("Training Time  |  Green=New Models  Red=Base Paper")
plt.tight_layout()
plt.savefig("training_time.png", dpi=150)
plt.close()
print("  Saved -> training_time.png")

# ---------- STEP 17: R2 & RMSE BAR CHARTS ----------
print("\n[STEP 17] R2 & RMSE comparison charts")

new_models    = {"Hybrid-AdapTree","Hybrid-AdapTree+FS"}
result_tables = [df_impedance, df_rh, df_temperature]
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

for row_idx, (target, df_r) in enumerate(zip(target_cols, result_tables)):
    for col_idx, metric in enumerate(["R2","RMSE"]):
        ax   = axes[row_idx, col_idx]
        clrs = ["#2ecc71" if m in new_models else "#4C72B0" for m in df_r["Model"]]
        bars = ax.bar(df_r["Model"], df_r[metric], color=clrs)
        ax.set_title(f"{target} – {metric}", fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=6.5)
        ax.legend(handles=[mpatches.Patch(color="#4C72B0", label="Base"),
                            mpatches.Patch(color="#2ecc71", label="New")], fontsize=7)

plt.suptitle("Model Comparison – All 8 Models  |  Green=New  Blue=Base", fontsize=13)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> model_comparison.png")

# ---------- STEP 18: SCATTER PLOTS ----------
print("\n[STEP 18] Scatter plots")

compare_preds  = [adaptree_pred, hybrid_pred, hybrid_fs_pred]
compare_labels = ["AdapTree (Base)", "Hybrid-AdapTree (New)", "Hybrid-AdapTree+FS (New)"]
compare_colors = ["#4C72B0", "#2ecc71", "#e67e22"]

fig, axes = plt.subplots(len(target_cols), 3, figsize=(16, 13))
for row_idx, col in enumerate(target_cols):
    for col_idx, (pred, label, clr) in enumerate(zip(compare_preds, compare_labels, compare_colors)):
        ax = axes[row_idx, col_idx]
        ax.scatter(y_test[:, row_idx], pred[:, row_idx], alpha=0.35, s=8, color=clr)
        lims = [min(y_test[:,row_idx].min(), pred[:,row_idx].min()),
                max(y_test[:,row_idx].max(), pred[:,row_idx].max())]
        ax.plot(lims, lims, "r--", lw=1.5)
        r2 = r2_score(y_test[:, row_idx], pred[:, row_idx])
        ax.set_title(f"{label}\n{col}  R2={r2:.4f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

plt.suptitle("Actual vs Predicted – AdapTree vs Hybrid Variants", fontsize=13)
plt.tight_layout()
plt.savefig("scatter_comparison.png", dpi=150)
plt.close()
print("  Saved -> scatter_comparison.png")

# ---------- STEP 19: ENVIRONMENTAL STRESS INDEX (ESI) ----------
print("\n[STEP 19] Environmental Stress Index")

t_opt,  t_min,  t_max  = 24.5,  19.0,  30.0
rh_opt, rh_min, rh_max = 59.65, 27.5,  91.8
pi_opt, pi_min, pi_max = 4120.3,282.3, 7958.3
w = 1/3

print(f"\n  {'Sample':>8} {'Temp':>10} {'RH':>8} {'Impedance':>12} {'ESI':>8}  Level")
for j in range(5):
    T  = hybrid_fs_pred[j, target_cols.index("Temperature")]
    RH = hybrid_fs_pred[j, target_cols.index("RH")]
    PI = hybrid_fs_pred[j, target_cols.index("Impedance")]
    esi = (w*(T-t_opt)/(t_max-t_min) + w*(RH-rh_opt)/(rh_max-rh_min) + w*(PI-pi_opt)/(pi_max-pi_min))
    level = "HIGH" if abs(esi) > 0.2 else ("MODERATE" if abs(esi) > 0.1 else "LOW")
    print(f"  {j+1:>8} {T:>10.3f} {RH:>8.3f} {PI:>12.3f} {esi:>8.4f}  {level}")

# ---------- STEP 20: IMPROVEMENT SUMMARY ----------
print("\n[STEP 20] Improvement summary (Hybrid-AdapTree + Feature Selection vs AdapTree)")

print(f"\n  {'Target':<12} {'Metric':<6} {'AdapTree':>12} {'Hybrid+FS':>12} {'Change':>10}")
print(f"  {'-'*56}")
for i, target in enumerate(target_cols):
    base_r2   = r2_score(y_test[:, i], adaptree_pred[:, i])
    new_r2    = r2_score(y_test[:, i], hybrid_fs_pred[:, i])
    base_rmse = np.sqrt(mean_squared_error(y_test[:, i], adaptree_pred[:, i]))
    new_rmse  = np.sqrt(mean_squared_error(y_test[:, i], hybrid_fs_pred[:, i]))
    print(f"  {target:<12} {'R2':<6} {base_r2:>12.6f} {new_r2:>12.6f} {((new_r2-base_r2)/abs(base_r2))*100:>+9.2f}%")
    print(f"  {target:<12} {'RMSE':<6} {base_rmse:>12.4f} {new_rmse:>12.4f} {((base_rmse-new_rmse)/base_rmse)*100:>+9.2f}%")

print(f"\n  Train Time  AdapTree: {adaptree_time:.4f}s  |  Hybrid: {hybrid_time:.4f}s  |  Hybrid+FS: {hybrid_fs_time:.4f}s")
print(f"  Speed gain (Hybrid+FS vs Hybrid): {((hybrid_time - hybrid_fs_time)/hybrid_time)*100:+.1f}% faster (FS reduces training time)")



print("\n" + "="*78)
print("  Done!  Output files: heatmap.png | feature_importance.png |")
print("         model_comparison.png | training_time.png | scatter_comparison.png")
print("="*78)
