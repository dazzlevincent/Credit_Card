import streamlit as st
import pymongo
import pandas as pd
import joblib
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler



# MongoDB Connection
MONGO_URI = "mongodb+srv://dazzlevincent:dazzle12@cluster0.61vaosq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = pymongo.MongoClient(MONGO_URI)

# Select Database & Collections
db = client["mydatabase"]
users_collection = db["users"]
transactions_collection = db["transactions"]

# Load Models
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Session State for authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "Login"  # Default page

# Function for Login
def login():
    """Handles user login"""
    st.subheader("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = users_collection.find_one({"username": username, "password": password})
        if user:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.page = "Fraud Detection"  # Redirect to fraud detection page
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials, please try again!")

# Function for Registration with auto-redirect to Login
import re  # Import regex module

import re  # Import regex module

import re
import streamlit as st

def register():
    """Handles user registration"""
    st.subheader("📝 Register")

    # New fields
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=18, max_value=100)

    if st.button("Register"):
        # **Validation Checks**
        if not first_name or not last_name:
            st.error("⚠️ First Name and Last Name are required!")
        elif not username:
            st.error("⚠️ Username cannot be empty!")
        elif users_collection.find_one({"username": username}):
            st.error("⚠️ Username already exists! Please choose another.")
        elif not re.match(r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{6,}$', username):
            st.error("⚠️ Username must be at least 6 characters long and contain both letters and numbers!")
        elif len(password) < 6:
            st.error("⚠️ Password must be at least 6 characters long!")
        elif password != confirm_password:
            st.error("⚠️ Passwords do not match!")
        else:
            user_data = {
                "first_name": first_name,
                "last_name": last_name,
                "username": username,
                "password": password,
                "gender": gender,
                "age": age
            }
            users_collection.insert_one(user_data)
            
            # 🎉 Show success notification
            st.toast("✅ Registered successfully!", icon="🎉")  
            st.success("🎉 Registration Successful! You can now proceed to Login.")

            # Add "Proceed to Login" button
            




# Function to Save Transactions
def save_transaction_to_db(transaction):
    """Stores a transaction in MongoDB under the logged-in username"""
    transaction["username"] = st.session_state.username
    transactions_collection.insert_one(transaction)

# Function to Fetch Transactions
def get_transactions_from_db():
    """Fetches transaction history for the logged-in user"""
    return list(transactions_collection.find({"username": st.session_state.username}, {"_id": 0}))

# Fraud Detection Function
# Fraud Detection Function
def fraud_detection():
    """Handles credit card fraud detection and stores transactions"""
    st.subheader("💳 Enter Credit Card Transaction Details")

    # Numeric Inputs
    amt = st.number_input("Amount", min_value=1.0, step=0.1, format="%.2f")
    lat = st.number_input("Latitude", format="%.6f")
    long = st.number_input("Longitude", format="%.6f")
    city_pop = st.number_input("City Population", min_value=1, step=1)
    unix_time = st.number_input("Unix Time", min_value=1, step=1)
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")

    # Categorical Inputs
    merchant = st.selectbox("Merchant",[
    "fraud_Koepp-Witting",
    "fraud_Bogisich-Homenick",
    "fraud_Bechtelar-Rippin",
    "fraud_Kilback Group",
    "fraud_Gottlieb-Hansen",
    "fraud_Ankunding LLC",
    "fraud_Reynolds-Schinner",
    "fraud_Kuhic LLC",
    "fraud_Kerluke-Abshire",
    "fraud_Torphy-Goyette",
    "fraud_Ernser-Feest",
    "fraud_Kutch, Steuber and Gerhold",
    "fraud_Streich, Hansen and Veum",
    "fraud_Heller-Langosh",
    "fraud_Abbott-Rogahn",
    "fraud_Reilly, Heaney and Cole",
    "fraud_Rowe, Batz and Goodwin",
    "fraud_Schmeler, Bashirian and Price",
    "fraud_Turner and Sons",
    "fraud_Eichmann, Bogan and Rodriguez",
    "fraud_Beier LLC",
    "fraud_Heidenreich Group",
    "fraud_Marquardt, Mayert and Nikolaus",
    "fraud_Bruen Ltd.",
    "fraud_Schroeder Inc.",
    "fraud_Lindgren, Hegmann and Crona",
    "fraud_Pollich, Schaefer and Zemlak",
    "fraud_Klein Inc.",
    "fraud_Hudson and Sons",
    "fraud_Boyle, Cremin and Kuhic",
    "fraud_Connelly, Hartmann and Kuhlman",
    "fraud_Deckow-Osinski",
    "fraud_Bauch LLC",
    "fraud_Weissnat LLC",
    "fraud_Koss Inc.",
    "fraud_Prohaska Ltd.",
    "fraud_Hansen, Stracke and Lueilwitz",
    "fraud_Morissette Group",
    "fraud_Bergstrom, Ritchie and Kertzmann",
    "fraud_Kub and Sons",
    "fraud_Abshire, Grimes and Schuster",
    "fraud_Schmidt, Green and Cartwright",
    "fraud_Crist and Sons",
    "fraud_Gaylord Ltd.",
    "fraud_Luettgen Inc.",
    "fraud_Koepp Group",
    "fraud_Kub, Frami and Roob",
    "fraud_Medhurst, Abbott and Kihn",
    "fraud_Davis, Schultz and Dickinson",
    "fraud_Swift, Harber and Sporer"
]

)
    category = st.selectbox("Category",[
    "grocery_pos",
    "misc_net",
    "food_dining",
    "personal_care",
    "shopping_net",
    "misc_pos",
    "shopping_pos",
    "home",
    "gas_transport",
    "entertainment",
    "health_fitness",
    "travel",
    "kids_pets",
    "education",
    "auto",
    "subscription",
    "digital_services",
    "electronics",
    "clothing_accessories",
    "luxury",
    "pharmacy",
    "sports",
    "books_magazines",
    "charity",
    "furniture",
    "hotel",
    "mobile_apps",
    "office_supplies",
    "outdoor",
    "utilities",
    "insurance",
    "legal_services",
    "music_streaming",
    "online_gaming",
    "parking",
    "public_transport",
    "rideshare",
    "software",
    "telecom",
    "toys_games",
    "video_streaming"
]
)
    gender = st.selectbox("Gender", ["M", "F"])
    job = st.selectbox("Job", [
    "Economist",
    "Broadcast presenter",
    "Freight forwarder",
    "Social research officer, government",
    "Environmental consultant",
    "Claims inspector/assessor",
    "Air cabin crew",
    "Engineer, production",
    "Scientist, physiological",
    "Corporate investment banker",
    "Software engineer",
    "Data analyst",
    "Machine learning engineer",
    "Cybersecurity specialist",
    "Blockchain developer",
    "Doctor",
    "Nurse",
    "Pharmacist",
    "Civil engineer",
    "Mechanical engineer",
    "Electrical engineer",
    "Architect",
    "Graphic designer",
    "Marketing manager",
    "HR specialist",
    "Financial analyst",
    "Investment banker",
    "Actuary",
    "Teacher",
    "Professor",
    "Research scientist",
    "Lawyer",
    "Judge",
    "Police officer",
    "Firefighter",
    "Journalist",
    "Writer",
    "Photographer",
    "Pilot",
    "Air traffic controller",
    "Athlete",
    "Musician",
    "Artist",
    "Chef",
    "Event planner",
    "Real estate agent",
    "Entrepreneur",
    "Sales manager",
    "Customer service representative",
    "Supply chain manager",
    "Consultant"
]
)

    if st.button("🔍 Check for Fraud"):
        # **Check if all required fields are filled**
        if not all([amt, lat, long, city_pop, unix_time, merch_lat, merch_long, merchant, category, gender, job]):
            st.warning("⚠️ Please fill in all required fields before proceeding!")
            return  # Stop execution if any field is missing

        # Prepare DataFrame
        input_data = pd.DataFrame([{
            "merchant": merchant,
            "category": category,
            "amt": amt,
            "gender": gender,
            "lat": lat,
            "long": long,
            "city_pop": city_pop,
            "job": job,
            "unix_time": unix_time,
            "merch_lat": merch_lat,
            "merch_long": merch_long
        }])

        # Encode categorical values
        categorical_cols = ["merchant", "category", "gender", "job"]
        for col in categorical_cols:
            input_data[col] = label_encoders[col].transform(input_data[col])

        # Scale numerical values
        num_cols = ["amt", "lat", "long", "city_pop", "unix_time", "merch_lat", "merch_long"]
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        # Convert to numpy and reshape
        X_new = input_data.values.reshape(1, -1)
        y_pred = xgb_model.predict(X_new)

        result = "🚨 Fraudulent Transaction" if y_pred[0] == 1 else "✅ Non-Fraudulent Transaction"
        st.success(f"Prediction: {result}")

        # Save Transaction to MongoDB
        transaction = {
            "Merchant": merchant,
            "Category": category,
            "Amount": amt,
            "Gender": gender,
            "Latitude": lat,
            "Longitude": long,
            "City Population": city_pop,
            "Job": job,
            "Unix Time": unix_time,
            "Merchant Lat": merch_lat,
            "Merchant Long": merch_long,
            "Result": result
        }
        save_transaction_to_db(transaction)


# Display Transaction History
def display_transaction_history():
    """Displays transaction history for the logged-in user"""
    st.subheader(f"📜 Transaction History for {st.session_state.username}")

    transactions = get_transactions_from_db()
    if transactions:
        df = pd.DataFrame(transactions)
        df = df.rename(columns={"Result": "Fraud Prediction"})
        st.dataframe(df)
    else:
        st.write("🔍 No transactions found.")

# Streamlit Navigation using session state
st.title("🔍 Credit Card Fraud Detection")
if st.session_state.authenticated:
    menu = ["Fraud Detection", "Transaction History", "Logout"]
else:
    menu = ["Login", "Register"]

choice = st.sidebar.selectbox("📌 Menu", menu, index=menu.index(st.session_state.page))

if choice == "Login":
    login()
elif choice == "Register":
    register()
elif choice == "Fraud Detection":
    fraud_detection()
elif choice == "Transaction History":
    display_transaction_history()
elif choice == "Logout":
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.page = "Login"  # Redirect to login after logout
    st.success("✅ Logged out successfully!")
    st.rerun()



