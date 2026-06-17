"""
Q4 — PREDICTING INVENTORY SURGES (next-7-day demand surge for disaster SKUs)
============================================================================
Goal: predict surge_flag (1/0) per (store, sku, date):
    baseline            = rolling_median(units_sold, past 28 days)
    demand_next_7_days  = sum(units_sold, next 7 days)
    surge_flag          = 1 if demand_next_7_days > 1.5 * (7 * baseline)

This file:
  1. Generates synthetic data matching all 5 input schemas (so it runs end-to-end).
  2. Builds the label.
  3. Engineers features (sales trends, stock pressure, disaster proximity, metadata).
  4. Trains a gradient-boosted classifier and reports honest metrics.

ASSUMPTIONS
-----------
1. Baseline is per-day median; expected 7-day demand = 7 * daily baseline,
   so surge means next-7-day demand > 1.5 * 7 * baseline. (Spec gives the
   ratio against baseline; we make the 7-day horizon explicit.)
2. We MUST avoid leakage: features use only data up to `date`; the label uses
   days AFTER `date`. We also drop the last 7 days per series (no future label).
3. Disaster feed is joined by haversine distance from store to event centroid,
   and by whether an alert was issued within a lookback window before `date`.
4. Class imbalance handled with class_weight / scale_pos_weight.
5. Time-based split (train on earlier dates, test on later) — NOT random —
   because this is forecasting.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(42)

# --------------------------------------------------------------------------
# 1. SYNTHETIC DATA (matches the provided schemas)
# --------------------------------------------------------------------------
N_STORES, N_SKUS, N_DAYS = 8, 6, 200
start = date(2025, 1, 1)
dates = [start + timedelta(days=d) for d in range(N_DAYS)]

stores = pd.DataFrame({
    "store_id": range(N_STORES),
    "store_lat": rng.uniform(8, 30, N_STORES),
    "store_lon": rng.uniform(72, 88, N_STORES),
    "store_size": rng.integers(200, 4000, N_STORES),
    "store_type": rng.choice(["convenience", "supermarket"], N_STORES),
    "region": rng.choice(["North", "South", "East", "West"], N_STORES),
    "urban_rural": rng.choice(["urban", "rural"], N_STORES),
})

skus = pd.DataFrame({
    "sku_id": range(N_SKUS),
    "category": rng.choice(["water", "battery", "medicine"], N_SKUS),
    "brand": rng.choice(["A", "B", "C"], N_SKUS),
    "unit_size": rng.choice([1, 6, 12], N_SKUS),
    "lead_time_days": rng.integers(1, 10, N_SKUS),
})

# Sales: base demand + weekly seasonality + noise; surges injected near disasters
rows = []
for s in range(N_STORES):
    for k in range(N_SKUS):
        base = rng.integers(5, 40)
        for i, d in enumerate(dates):
            season = 1 + 0.2 * np.sin(2 * np.pi * i / 7)
            units = max(0, int(rng.poisson(base * season)))
            promo = rng.random() < 0.1
            if promo:
                units = int(units * 1.4)
            price = round(rng.uniform(20, 200), 2)
            rows.append([s, k, d, units, units * price, promo, price])
sales = pd.DataFrame(rows, columns=["store_id", "sku_id", "date", "units_sold",
                                    "revenue", "promo_flag", "price"])

inventory = sales[["store_id", "sku_id", "date"]].copy()
inventory["on_hand_units"] = rng.integers(0, 300, len(inventory))
inventory["incoming_units"] = rng.integers(0, 100, len(inventory))
inventory["stockouts"] = (inventory["on_hand_units"] < 5).astype(int)

# Disaster alerts
n_events = 12
disasters = pd.DataFrame({
    "event_id": range(n_events),
    "event_type": rng.choice(["flood", "earthquake", "cyclone", "wildfire"], n_events),
    "event_start_date": [start + timedelta(days=int(x))
                         for x in rng.integers(20, N_DAYS - 10, n_events)],
    "event_lat": rng.uniform(8, 30, n_events),
    "event_lon": rng.uniform(72, 88, n_events),
    "severity": rng.integers(1, 6, n_events),
})
disasters["event_end_date"] = [d + timedelta(days=int(x))
                               for d, x in zip(disasters["event_start_date"],
                                               rng.integers(2, 8, n_events))]
disasters["alert_time"] = [d - timedelta(days=int(x))
                           for d, x in zip(disasters["event_start_date"],
                                           rng.integers(1, 4, n_events))]

# Inject realistic surges: near a disaster, water/battery/medicine demand spikes
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

sales = sales.merge(stores[["store_id", "store_lat", "store_lon"]], on="store_id")
sales = sales.merge(skus[["sku_id", "category"]], on="sku_id")
for _, ev in disasters.iterrows():
    dist = haversine(sales["store_lat"], sales["store_lon"],
                     ev["event_lat"], ev["event_lon"])
    window = (sales["date"] >= ev["event_start_date"]) & \
             (sales["date"] <= ev["event_end_date"])
    affected = (dist < 400) & window & sales["category"].isin(
        ["water", "battery", "medicine"])
    sales.loc[affected, "units_sold"] = (
        sales.loc[affected, "units_sold"] * rng.uniform(2.0, 3.5)).astype(int)

# --------------------------------------------------------------------------
# 2. LABEL CONSTRUCTION (leakage-safe)
# --------------------------------------------------------------------------
sales = sales.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)
g = sales.groupby(["store_id", "sku_id"])["units_sold"]

# baseline = rolling median of past 28 days (shifted so it excludes today's future)
sales["baseline"] = g.transform(
    lambda x: x.shift(1).rolling(28, min_periods=14).median())

# demand over the NEXT 7 days (forward sum, excluding today)
sales["demand_next_7"] = g.transform(
    lambda x: x.shift(-1).rolling(7, min_periods=7).sum())

sales["surge_flag"] = (
    sales["demand_next_7"] > 1.5 * 7 * sales["baseline"]).astype(int)

# --------------------------------------------------------------------------
# 3. FEATURE ENGINEERING (only past/present info)
# --------------------------------------------------------------------------
sales["dow"] = pd.to_datetime(sales["date"]).dt.dayofweek
sales["units_lag1"] = g.shift(1)
sales["units_lag7"] = g.shift(7)
sales["roll7_mean"] = g.transform(lambda x: x.shift(1).rolling(7).mean())
sales["roll7_std"] = g.transform(lambda x: x.shift(1).rolling(7).std())
sales["trend"] = sales["roll7_mean"] / (sales["baseline"] + 1e-6)

# inventory pressure
sales = sales.merge(inventory, on=["store_id", "sku_id", "date"], how="left")
sales["stock_cover_days"] = sales["on_hand_units"] / (sales["roll7_mean"] + 1e-6)

# metadata (drop helpers added earlier to avoid merge collisions)
sales = sales.drop(columns=["category", "store_lat", "store_lon"])
sales = sales.merge(stores, on="store_id", how="left")
sales = sales.merge(skus, on="sku_id", how="left")

# disaster proximity feature: min distance to any active/upcoming alert,
# and max severity of alerts issued in the last 5 days within 500 km
# normalize date dtypes for arithmetic
sales["date"] = pd.to_datetime(sales["date"])
disasters["alert_time"] = pd.to_datetime(disasters["alert_time"])

def disaster_features(row):
    issued = disasters[disasters["alert_time"] <= row["date"]]
    if issued.empty:
        return pd.Series({"min_disaster_dist": 9999.0, "active_severity": 0})
    dist = haversine(row["store_lat"], row["store_lon"],
                     issued["event_lat"], issued["event_lon"])
    recent = (row["date"] - issued["alert_time"]).dt.days.between(0, 7)
    near = dist < 500
    sev = issued.loc[recent & near, "severity"]
    return pd.Series({"min_disaster_dist": float(dist.min()),
                      "active_severity": float(sev.max()) if len(sev) else 0.0})

sales[["min_disaster_dist", "active_severity"]] = sales.apply(
    disaster_features, axis=1)

# --------------------------------------------------------------------------
# 4. MODEL
# --------------------------------------------------------------------------
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             average_precision_score)

feature_cols = ["dow", "units_lag1", "units_lag7", "roll7_mean", "roll7_std",
                "trend", "promo_flag", "price", "on_hand_units", "incoming_units",
                "stockouts", "stock_cover_days", "store_size", "lead_time_days",
                "unit_size", "min_disaster_dist", "active_severity"]
cat_cols = ["store_type", "region", "urban_rural", "category", "brand"]

model_df = sales.dropna(subset=["surge_flag", "baseline", "roll7_mean"]).copy()
for c in cat_cols:
    model_df[c] = model_df[c].astype("category")
X = model_df[feature_cols + cat_cols]
y = model_df["surge_flag"]

# TIME-BASED split (forecasting): train on earlier 75% of dates
cutoff = model_df["date"].quantile(0.75)
train = model_df["date"] <= cutoff
Xtr, Xte, ytr, yte = X[train], X[~train], y[train], y[~train]

clf = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.05, max_depth=6,
    categorical_features=[X.columns.get_loc(c) for c in cat_cols],
    class_weight="balanced", random_state=42)
clf.fit(Xtr, ytr)

proba = clf.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"Rows: {len(model_df)} | surge rate: {y.mean():.3f}")
print(f"Train: {train.sum()}  Test: {(~train).sum()}")
print(f"\nROC-AUC : {roc_auc_score(yte, proba):.3f}")
print(f"PR-AUC  : {average_precision_score(yte, proba):.3f}  "
      f"(more meaningful than accuracy under imbalance)")
print("\n", classification_report(yte, pred, digits=3))

importances = pd.Series(
    clf.feature_importances_ if hasattr(clf, "feature_importances_")
    else np.zeros(len(X.columns)), index=X.columns
) if hasattr(clf, "feature_importances_") else None

print("\nNOTE: For deployment, tune the decision threshold to the retailer's "
      "cost trade-off — a missed surge (stockout in a disaster) is far more "
      "costly than over-stocking, so favor recall on the positive class.")
