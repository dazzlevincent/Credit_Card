
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

xgb_model = joblib.load("xgboost_model.pkl")

df = pd.read_csv("sample11.csv")  

# drop_cols = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "city", "state", "zip", "dob", "trans_num"]
# df = df.drop(columns=drop_cols, errors="ignore")


# categorical_cols = ["merchant", "category", "gender", "job"]
# for col in categorical_cols:
#     df[col] = LabelEncoder().fit_transform(df[col])

# num_cols = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
# scaler = StandardScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])
# print(df)
# joblib.dump(scaler, "scaler.pkl")  # Save the fitted scaler

# X = df.drop(columns=["is_fraud"], errors="ignore")
# print(df)
# y_pred_xgb = xgb_model.predict(X)

# print("XGBoost_Prediction")
# print(y_pred_xgb)

# res = y_pred_xgb[0]
# print(res)
# if res == 0:
#     print("non fraud")
# print(res)
# if res==0:
#     print("non fraud")
# else:
#     print("fraud")


xgb_model = joblib.load("xgboost_model.pkl")
   
df = pd.read_csv("book2.csv")  

drop_cols = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "city", "state", "zip", "dob", "trans_num"]
df = df.drop(columns=drop_cols, errors="ignore")

categorical_cols = ["merchant", "category", "gender", "job"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
print(df)
num_cols = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
   
X = df.drop(columns=["is_fraud"], errors="ignore")

y_pred_xgb = xgb_model.predict(X)

print("XGBoost_Prediction")
print(y_pred_xgb)
       
       
res=y_pred_xgb[0]
print(res)
if res==0:
    print("non fraud")
else:
    print("fraud")