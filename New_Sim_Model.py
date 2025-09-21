# # Assume we predict daily log returns (y_pred)

# import numpy as np
# import pandas as pd
# import pandas as pd
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from xgboost import XGBRegressor

# df = pd.read_csv("sim_daily_dataset.csv")
# df['date'] = pd.to_datetime(df['date'])
# df = df.sort_values(['company_id','date'])

# # 1) Build next-day target (log return)
# df['y'] = df.groupby('company_id')['realized_log_return'].shift(-1)
# train = df.dropna(subset=['y'])

# # 2) Pick features (same set you’ll expose as sliders)
# num_cols = [
#   "overall_market_sentiment","fii_flows","dii_flows","global_market_cues",
#   "inr_usd_delta","crude_oil_delta","company_size","analyst_rating_change",
#   "earnings_announcement"
# ]
# cat_cols = ["sector","market_cap_bucket","major_news","insider_activity","predefined_global_shock"]

# X = train[num_cols + cat_cols]
# y = train['y']

# # 3) Fit XGBoost (or your choice) and predict next-day log returns
# # (pipeline code omitted here for brevity — same as earlier)
# pre = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#     ]
# )

# # (Optional) monotonic constraints: same order as num_cols
# # e.g., sentiment (+), FII (+), DII (+), global_cues (+), inr_usd_delta (often -), crude (-),
# # size (?), analyst_change (+), earnings (+)

# # Only constrain two numeric features you’re sure about:
# num_mono = [ +1,  # overall_market_sentiment
#              0,   # fii_flows
#              0,   # dii_flows
#              0,   # global_market_cues (can flip in crises)
#              0,   # inr_usd_delta (exporters vs importers)
#              0,   # crude_oil_delta (sector-dependent)
#              0,   # company_size
#              +1,  # analyst_rating_change
#              0 ]  # earnings_announcement (surprise can be ±)
# # All one-hot categorical columns -> 0


# # Fit preprocessor to learn one-hot sizes
# pre.fit(X)  # X = df[num_cols+cat_cols]

# # Count columns
# n_num = pre.named_transformers_["num"].mean_.shape[0]
# n_cat = pre.named_transformers_["cat"].get_feature_names_out().shape[0]

# # Build full constraint vector: numeric signs + zeros for all one-hot columns
# full_mono = np.concatenate([num_mono, np.zeros(n_cat-2, dtype=int)])
# mono_str  = "(" + ",".join(map(str, full_mono.tolist())) + ")"

# model=XGBRegressor(
#   n_estimators=800,
#   learning_rate=0.03,
#   max_depth=4,
#   min_child_weight=4,
#   subsample=0.8,
#   colsample_bytree=0.8,
#   reg_alpha=0.0,
#   reg_lambda=2.0,
#   tree_method="hist",
#   objective="reg:absoluteerror"   # or "reg:pseudohubererror" if available
# )

# pipe = Pipeline([("pre", pre), ("xgb", model)])

# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
# pipe.fit(X_tr, y_tr)

# print("R^2 (holdout):", pipe.score(X_te, y_te))
# y_pred = pipe.predict(X_te)
# from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score
# import numpy as np

# mae = mean_absolute_error(y_te, y_pred)
# rmse = root_mean_squared_error(y_te, y_pred)
# directional_acc = accuracy_score(y_te > 0, y_pred > 0)
# corr = np.corrcoef(y_te, y_pred)[0,1]

# print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Directional Acc: {directional_acc:.2%}, Corr: {corr:.2f}")


# def simulate_ohlc(dates, start_price, y_pred, vol=0.02):
#     prices = []
#     open_p = start_price
    
#     for r in y_pred:
#         close_p = open_p * np.exp(r)   # log-return
#         high_p  = max(open_p, close_p) * (1 + np.random.uniform(0, vol))
#         low_p   = min(open_p, close_p) * (1 - np.random.uniform(0, vol))
        
#         prices.append({
#             "date": dates[len(prices)],
#             "open": open_p,
#             "high": high_p,
#             "low": low_p,
#             "close": close_p
#         })
#         open_p = close_p
#     return pd.DataFrame(prices)
# if __name__=="__main__":
#     y_pred_test=pipe.predict(X_te[0])
#     dates = pd.date_range('2025-07-01', periods=10, freq='B')
#     start_price = 215.40
#     ohlc=simulate_ohlc('2025-07-01',start_price,y_pred_test)

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score
from xgboost import XGBRegressor

# ---------- Load & target ----------
df = pd.read_csv("sim_daily_dataset.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['company_id','date'])
df['y'] = df.groupby('company_id')['realized_log_return'].shift(-1)
train = df.dropna(subset=['y'])

num_cols = [
  "overall_market_sentiment","fii_flows","dii_flows","global_market_cues",
  "inr_usd_delta","crude_oil_delta","company_size","analyst_rating_change",
  "earnings_announcement"
]
cat_cols = ["sector","market_cap_bucket","major_news","insider_activity","predefined_global_shock"]

X = train[num_cols + cat_cols]
y = train['y']

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---- optional: light monotonicity (sentiment+, analyst+) ----
pre.fit(X)  # to know the expanded sizes
n_num = pre.named_transformers_["num"].mean_.shape[0]
n_cat = pre.named_transformers_["cat"].get_feature_names_out().shape[0]

# num_mono = np.array([+1, 0, 0, 0, 0, 0, 0, +1, 0], dtype=int)  # only 2 constraints
# full_mono = np.concatenate([num_mono, np.zeros(n_cat, dtype=int)])
# mono_str  = "(" + ",".join(map(str, full_mono.tolist())) + ")"

model = XGBRegressor(
  n_estimators=800, learning_rate=0.03, max_depth=4, min_child_weight=4,
  subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, reg_alpha=0.0,
  tree_method="hist", objective="reg:absoluteerror",
  
)

pipe = Pipeline([("pre", pre), ("xgb", model)])

# time-ordered split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
pipe.fit(X_tr, y_tr)

print("R^2 (holdout):", pipe.score(X_te, y_te))
y_pred = pipe.predict(X_te)

mae  = mean_absolute_error(y_te, y_pred)
rmse = root_mean_squared_error(y_te, y_pred )
dir_acc = accuracy_score(y_te > 0, y_pred > 0)
corr = np.corrcoef(y_te, y_pred)[0,1]
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, Directional Acc: {dir_acc:.2%}, Corr: {corr:.2f}")

# ---------- OHLC builder ----------
def make_ohlc_from_returns(dates, start_price, returns_log, base_vol=0.012):
    """dates: sequence of Timestamps; returns_log: array of log-returns"""
    ohlc = []
    open_p = float(start_price)
    for d, r in zip(dates, returns_log):
        close_p = open_p * np.exp(r)
        intra = abs(r) + base_vol*0.5  # wick size ~ move + base vol
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, intra))
        low_p  = min(open_p, close_p)  * (1 - np.random.uniform(0, intra))
        ohlc.append({"date": d, "open": open_p, "high": high_p, "low": low_p, "close": close_p})
        open_p = close_p
    return pd.DataFrame(ohlc)

# ---------- Build a realistic test chart for ONE company ----------
test_df = train.iloc[X_te.index]  # rows aligned to test
# pick a company with enough rows
cid = test_df['company_id'].value_counts().idxmax()
test_one = test_df[test_df['company_id'] == cid].copy()

# features for that slice, then predict
X_one = test_one[num_cols + cat_cols]
yhat_one = pipe.predict(X_one)

# start price = last known close BEFORE this slice
all_company = df[df['company_id'] == cid].sort_values('date')
start_idx = all_company.index.get_loc(test_one.index[0])  # position in full company series
start_price = all_company.iloc[start_idx - 1]['close'] if start_idx > 0 else all_company.iloc[0]['open']

# ohlc = make_ohlc_from_returns(test_one['date'], start_price, yhat_one, base_vol=0.012)
# print(ohlc)
import numpy as np
import matplotlib.pyplot as plt

def plot_candles(ohlc_df, title="Candlestick"):
    # ohlc_df: columns = ['date','open','high','low','close']
    ohlc_df = ohlc_df.sort_values('date').reset_index(drop=True)
    xs = np.arange(len(ohlc_df))
    w = 0.6

    fig, ax = plt.subplots(figsize=(10,4))
    for i, r in ohlc_df.iterrows():
        # wick
        ax.plot([xs[i], xs[i]], [r['low'], r['high']])
        # body
        lower = min(r['open'], r['close'])
        height = abs(r['close'] - r['open'])
        ax.add_patch(plt.Rectangle((xs[i]-w/2, lower), w, max(height, 1e-9)))
    ax.set_xticks(xs[::max(1, len(xs)//10)])
    ax.set_xticklabels(ohlc_df['date'].dt.strftime('%Y-%m-%d').iloc[::max(1, len(xs)//10)],
                       rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()




import numpy as np, pandas as pd
from pandas.tseries.offsets import BDay

# --- 1) build future trading dates ---
def future_business_days(start_date, days=88):
    dts = pd.bdate_range(start_date + BDay(1), periods=days)
    return dts

# --- 2) expand a scenario into a feature panel ---
def make_future_features(
    company_meta,              # dict: {'company_id', 'sector', 'market_cap_bucket', 'company_size'}
    start_date, horizon=88,
    mode="hold",               # "hold" | "trajectory"
    controls=None,             # dict of scalars or per-day arrays
    events=None                # list of {'date':..., 'field':..., 'value':...}
):
    """Returns a DataFrame with the same feature columns your model expects."""
    dates = future_business_days(pd.Timestamp(start_date), horizon)
    n = len(dates)

    # defaults if user omits anything
    if (mode=="hold"):
        def _val(name, default):
            v = (controls or {}).get(name, default)
            if np.isscalar(v): return np.repeat(v, n)
            v = np.asarray(v)
            if len(v) != n: raise ValueError(f"{name} length {len(v)} != horizon {n}")
            return v
    else:
        def _val(name, default):
            v = (controls or {}).get(name, default)
            if np.isscalar(v): return np.repeat(v, n)
            v = np.asarray(v)
            if len(v) != n: raise ValueError(f"{name} length {len(v)} != horizon {n}")
            return v       


    # numeric features (daily scale)
    data = dict(
        overall_market_sentiment = _val("overall_market_sentiment", 0.0),
        fii_flows                 = _val("fii_flows", 0.0),         # crores per day
        dii_flows                 = _val("dii_flows", 0.0),
        global_market_cues        = _val("global_market_cues", 0.0),
        inr_usd_delta             = _val("inr_usd_delta", 0.0),     # e.g. +0.002 = +0.2%
        crude_oil_delta           = _val("crude_oil_delta", 0.0),   # e.g. -0.01 = -1%
        earnings_announcement     = _val("earnings_announcement", 0),    # 0/1
        analyst_rating_change     = _val("analyst_rating_change", 0)     # -2..+2
    )

    # categoricals (single choice each day; can be arrays)
    cat = dict(
        sector                = np.repeat(company_meta['sector'], n),
        market_cap_bucket     = np.repeat(company_meta['market_cap_bucket'], n),
        major_news            = _val("major_news", "none"),
        insider_activity      = _val("insider_activity", "none"),
        predefined_global_shock = _val("predefined_global_shock", "none"),
    )

    # optional event overrides (date-stamped mutations)
    if events:
        df_tmp = pd.DataFrame({**data, **cat}, index=dates).reset_index().rename(columns={'index':'date'})
        for ev in events:
            ix = df_tmp['date'] == pd.Timestamp(ev['date'])
            df_tmp.loc[ix, ev['field']] = ev['value']
        df_tmp['company_id'] = company_meta['company_id']
        df_tmp['company_name'] = company_meta.get('company_name', company_meta['company_id'])
        df_tmp['company_size'] = company_meta['company_size']
        return df_tmp

    # assemble
    out = pd.DataFrame({**data, **cat})
    out.insert(0, 'date', dates)
    out['company_id'] = company_meta['company_id']
    out['company_name'] = company_meta.get('company_name', company_meta['company_id'])
    out['company_size'] = company_meta['company_size']
    return out

# --- 3) predict returns and make OHLC ---
def make_ohlc_from_returns(dates, start_price, returns_log, base_vol=0.012):
    ohlc, open_p = [], float(start_price)
    for d, r in zip(pd.to_datetime(dates), returns_log):
        close_p = open_p * np.exp(r)
        intra = abs(r) + base_vol*0.5
        high_p = max(open_p, close_p) * (1 + np.random.uniform(0, intra))
        low_p  = min(open_p, close_p)  * (1 - np.random.uniform(0, intra))
        ohlc.append({"date": d, "open": open_p, "high": high_p, "low": low_p, "close": close_p})
        open_p = close_p
    return pd.DataFrame(ohlc)

def simulate_path(pipe, company_meta, last_known_close, start_date, horizon=88, controls=None, events=None, base_vol=0.012):
    # Build the same columns the model expects:
    num_cols = [
      "overall_market_sentiment","fii_flows","dii_flows","global_market_cues",
      "inr_usd_delta","crude_oil_delta","company_size","analyst_rating_change","earnings_announcement"
    ]
    cat_cols = ["sector","market_cap_bucket","major_news","insider_activity","predefined_global_shock"]

    future_df = make_future_features(company_meta, start_date, horizon, mode="hold",controls=controls, events=events)
    # order columns for the pipeline
    X_fut = future_df[num_cols + cat_cols]
    # predict daily log-returns
    rhat = pipe.predict(X_fut)
    # turn into OHLC path
    ohlc = make_ohlc_from_returns(future_df['date'], last_known_close, rhat, base_vol=base_vol)
    return ohlc, future_df, rhat
company_meta = {
  "company_id": "FIN-LC-01",
  "company_name": "FS-LC01",
  "sector": "Financial Services",
  "market_cap_bucket": "Large Cap",
  "company_size": 85
}
# controls = {
#   "overall_market_sentiment": 0.4,
#   "fii_flows": 800, "dii_flows": 300,
#   "global_market_cues": 0.2,
#   "inr_usd_delta": -0.001,         # +0.1% per day
#   "crude_oil_delta": -0.004,      # -0.4% per day
#   "earnings_announcement": 0,
#   "analyst_rating_change": 0,
#   "major_news": "none",
#   "insider_activity": "none",
#   "predefined_global_shock": "none"
# }
n = 88
controls = {
  "overall_market_sentiment": np.linspace(0.0, 0.6, n),  # ramps up
  "fii_flows": np.repeat(500, n),
  "global_market_cues": np.sin(np.linspace(0, 2*np.pi, n))*0.2,
  "major_news": ["none"]*n,
  "insider_activity": ["none"]*n,
  "predefined_global_shock": ["none"]*n,
  "earnings_announcement": [0]*n,
  "analyst_rating_change": [0]*n,
  "inr_usd_delta": [0.0]*n,
  "crude_oil_delta": [0.0]*n,
}

last_close   = 112.3           # from your latest historical row
start_date   = pd.Timestamp("2025-06-27")
ohlc, fut, rhat = simulate_path(pipe, company_meta, last_close, start_date, horizon=88, controls=controls, base_vol=0.012)
# -> plot ohlc
plot_candles(ohlc)