# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:56:28 2025

@author: ASUS
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

plt.rcParams["figure.dpi"] = 300
sns.set(style="whitegrid")

# ── 讀資料 ──
df = pd.read_csv(r"C:\Users\ASUS\Desktop\fenis\etm_results.csv").dropna()
df["disp_max_um"] = df["disp_max"] * 1e6
df["sigma_vm_MPa"] = df["sigma_vm_max"] / 1e6
df["mat_pair"] = df["mat_left"] + "-" + df["mat_right"]

# ── 編碼材料類別 ──
enc = LabelEncoder()
df["mat_left_enc"] = enc.fit_transform(df["mat_left"])
df["mat_right_enc"] = enc.fit_transform(df["mat_right"])

X = df[["J_in", "h_conv", "mat_left_enc", "mat_right_enc"]]

# ── 模型訓練與繪圖函數 ──
def train_and_plot(y, name, unit, file_suffix):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                             max_depth=4, subsample=0.8, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    R2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)

    # ── Pred vs FEM ──
    y_test_rounded = np.round(y_test, 2)
    y_pred_rounded = np.round(y_pred, 2)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test_rounded, y_pred_rounded, alpha=0.6)
    lims = [min(y_test_rounded.min(), y_pred_rounded.min()), 
            max(y_test_rounded.max(), y_pred_rounded.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel(f"FEM {name} ({unit})")
    plt.ylabel(f"Predicted {name} ({unit})")
    plt.title(f"Fig. 7a ({file_suffix})  XGBoost Pred vs FEM\n$R^2$={R2:.3f}, MAE={MAE:.2f}")
    plt.tight_layout()
    plt.savefig(f"Fig7a_{file_suffix}_Pred_vs_FEM.png")
    plt.close()

    # ── Feature Importance (手動畫，保留兩位有效數，不用科學記號) ──
    plt.figure(figsize=(7, 4))
    ax = plt.gca()

    importance_dict = model.get_booster().get_score(importance_type="gain")
    importance_df = pd.DataFrame(importance_dict.items(), columns=["Feature", "Gain"])
    importance_df = importance_df.sort_values(by="Gain", ascending=False)

    # 保留兩位小數且不用科學記號
    importance_df["Gain_fmt"] = importance_df["Gain"].apply(lambda x: float(f"{x:.2f}"))

    # 繪製 bar 圖
    ax.barh(importance_df["Feature"], importance_df["Gain_fmt"], color="steelblue")
    for i, (feature, gain) in enumerate(zip(importance_df["Feature"], importance_df["Gain_fmt"])):
        ax.text(gain + 1, i, f"{gain:.2f}", va="center")

    ax.invert_yaxis()
    ax.set_xlabel("Importance score")
    ax.set_title(f"Fig. 7b ({file_suffix})  Feature Importance (XGBoost Gain)")
    plt.tight_layout()
    plt.savefig(f"Fig7b_{file_suffix}_Feature_Importance.png")
    plt.close()

# ── 執行三個目標變數的模型 ──
train_and_plot(df["T_max"], "T_max", "K", "Tmax")
train_and_plot(df["disp_max_um"], "disp_max", "µm", "disp")
train_and_plot(df["sigma_vm_MPa"], "sigma_vm_max", "MPa", "stress")

print("✅ 圖片已存為 Fig7a/b_*.png，顯示格式已調整為兩位小數非科學記號")


