import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the original dataset used for training
df = pd.read_csv("balanced_dataset.csv")  # Use the dataset with all possible categories

# List of categorical columns
categorical_cols = ["merchant", "category", "gender", "job"]

# Dictionary to store label encoders
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Ensure the column is string
    df[col] = np.where(df[col].isin(le.classes_), le.transform(df[col]), 0)  # Assign default label for unseen categories

    label_encoders[col] = le  # Store the encoder

# Save the updated label encoders
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Updated label_encoders.pkl has been saved successfully!")

print("✅ Updated label_encoders.pkl has been saved successfully!")
