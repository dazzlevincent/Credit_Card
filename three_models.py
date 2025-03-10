import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

df = pd.read_csv("balanced_dataset.csv")  

drop_cols = ["Unnamed: 0", "trans_date_trans_time", "cc_num", "first", "last", "street", "city", "state", "zip", "dob", "trans_num"]
df = df.drop(columns=drop_cols, errors="ignore")

categorical_cols = ["merchant", "category", "gender", "job"]
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

num_cols = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# rf tr
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_acc:.4f}")
joblib.dump(rf, "random_forest_model.pkl")

#xg tr
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_acc:.4f}")
joblib.dump(xgb, "xgboost_model.pkl")

X_normal = df[df['is_fraud'] == 0][num_cols].values
X_fraud = df[df['is_fraud'] == 1][num_cols].values
X_normal = np.reshape(X_normal, (X_normal.shape[0], 1, X_normal.shape[1]))
X_fraud = np.reshape(X_fraud, (X_fraud.shape[0], 1, X_fraud.shape[1]))
X_train_lstm, X_val_lstm = train_test_split(X_normal, test_size=0.2, random_state=42)

input_dim = X_train_lstm.shape[2]
inputs = Input(shape=(1, input_dim))
encoded = LSTM(32, activation='relu', return_sequences=False)(inputs)
encoded = Dropout(0.2)(encoded)
decoded = RepeatVector(1)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(input_dim))(decoded)

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')

history = autoencoder.fit(X_train_lstm, X_train_lstm, 
                          epochs=50, 
                          batch_size=32, 
                          validation_data=(X_val_lstm, X_val_lstm), 
                          verbose=1)

autoencoder.save("lstm_autoencoder.h5")

X_combined = np.concatenate([X_normal, X_fraud], axis=0)
reconstructions = autoencoder.predict(X_combined)
mse = np.mean(np.power(X_combined - reconstructions, 2), axis=(1, 2))

y_true = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_fraud))])
fpr, tpr, thresholds = roc_curve(y_true, mse)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred_lstm = (mse > optimal_threshold).astype(int)


print("opt thresh:", optimal_threshold)
print(classification_report(y_true, y_pred_lstm))

# Plot cm
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

plot_confusion_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_xgb, "XGBoost Confusion Matrix")
plot_confusion_matrix(y_true, y_pred_lstm, "LSTM Autoencoder Confusion Matrix")


print("\nModel Accuracy Comparison:")
print(f"Random Forest: {rf_acc:.4f}")
print(f"XGBoost: {xgb_acc:.4f}")
print(f"LSTM Autoencoder: {accuracy_score(y_true, y_pred_lstm):.4f}")

lstm_acc = accuracy_score(y_true, y_pred_lstm)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(np.trapz(tpr, fpr)))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid()
plt.show()

accuracies = {
    "Random Forest": rf_acc,
    "XGBoost": xgb_acc,
    "LSTM Autoencoder": lstm_acc
}

plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'], alpha=0.7)
plt.ylim(0, 1)  #range 0-1 acu
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Show accuracy values on bars
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')

plt.show()
