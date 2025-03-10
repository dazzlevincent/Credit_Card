import pandas as pd
import joblib
import numpy as np

xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl") 
label_encoders = joblib.load("label_encoders.pkl")  

df_new = pd.read_csv("udatatest.csv")

drop_cols = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street",
             "city", "state", "zip", "dob", "trans_num"]
df_new = df_new.drop(columns=drop_cols, errors="ignore")

categorical_cols = ["merchant", "category", "gender", "job"]
for col in categorical_cols:
    if col in df_new:
        df_new[col] = label_encoders[col].transform(df_new[col])

num_cols = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
df_new[num_cols] = scaler.transform(df_new[num_cols])

X_new = df_new.drop(columns=["is_fraud"], errors="ignore")  

X_new = X_new.values.reshape(1, -1)

y_pred_val = xgb_model.predict_proba(X_new)[:, 1] 


y_pred = int(y_pred_val[0] > 0.5)  

print(f"Predicted Label: {'Fraud' if y_pred == 1 else 'Not Fraud'}")
