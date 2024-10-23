import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError
from datetime import datetime
import mysql.connector

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Function to preprocess the text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Connect to the MySQL database on XAMPP
def connect_to_db():
    conn = mysql.connector.connect(
        host="localhost",        # XAMPP MySQL server
        user="your_username",    # Replace with your MySQL username
        password="your_password",# Replace with your MySQL password
        database="SpamRepositoryDB" # Replace with your MySQL database name
    )
    return conn

# Function to log classified messages into the database
def log_classification_to_db(message, prediction, confidence):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Insert the message, prediction, and confidence into the SpamRepository table
    cursor.execute("""
        INSERT INTO SpamRepository (MessageText, SpamLabel, DateAdded, MessageType)
        VALUES (%s, %s, %s, %s)
    """, (message, prediction == 1, datetime.now(), 'Detected Spam' if prediction == 1 else 'Not Spam'))

    conn.commit()
    conn.close()

# Function to check if a message exists in the spam repository
def check_in_spam_repository(message):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Check if the message exists in the SpamRepository table
    cursor.execute("""
        SELECT COUNT(*) FROM SpamRepository
        WHERE MessageText = %s AND SpamLabel = 1
    """, (message,))

    result = cursor.fetchone()[0]
    conn.close()

    return result > 0

# Load the vectorizer and model separately
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("The required files (vectorizer.pkl or model.pkl) were not found. Please ensure they exist in the application directory.")
except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")

# Streamlit app title with background
st.markdown(
    """
    <style>
    .main-header {
        font-size:48px;
        text-align:center;
        color:white;
        background-color:#6A5ACD;
        padding:20px;
        border-radius:15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="main-header">📱 SMS Spam Classifier</div>', unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Detect whether an SMS message is <strong>Spam</strong> or <strong>Not Spam</strong>.</p>", unsafe_allow_html=True)

# Input text box for the user with a customized style
st.markdown(
    """
    <style>
    .input-text-area textarea {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #6A5ACD;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
input_sms = st.text_area(
    "Enter the message below:",
    height=150,
    placeholder="Type your SMS message here...",
    key="input",
    help="Type the SMS message you want to classify.",
    label_visibility="collapsed"
)

# Predict button with custom style
st.markdown(
    """
    <style>
    .custom-button {
        display: flex;
        justify-content: center;
    }
    .custom-button button {
        background-color: #6A5ACD;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        cursor: pointer;
        transition: 0.3s;
    }
    .custom-button button:hover {
        background-color: #836FFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="custom-button">', unsafe_allow_html=True)
if st.button('Predict 🚀'):
    if input_sms.strip() == "":
        st.warning("Please enter an SMS message to classify.")
    else:
        try:
            # 1. Preprocess the input text
            transformed_sms = transform_text(input_sms)

            # 2. Check if the message exists in the spam repository
            is_in_spam_db = check_in_spam_repository(transformed_sms)

            # 3. If the message is already flagged as spam in the database
            if is_in_spam_db:
                st.markdown(
                    f"<div style='text-align: center; color: white; background-color: #FF4B4B; padding: 15px; border-radius: 15px;'>"
                    f"<h2>🚨 Spam (Detected from Database) 🚨</h2>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                # 4. Vectorize the transformed text
                vectorized_sms = vectorizer.transform([transformed_sms])

                # 5. Make the prediction using the loaded model and get confidence score
                prediction = model.predict(vectorized_sms)[0]
                confidence = model.predict_proba(vectorized_sms)[0][prediction] * 100  # Confidence percentage

                # 6. Display the result with enhanced visuals and confidence score
                if prediction == 1:
                    st.markdown(
                        f"<div style='text-align: center; color: white; background-color: #FF4B4B; padding: 15px; border-radius: 15px;'>"
                        f"<h2>🚨 Spam 🚨</h2>"
                        f"<p>Confidence: {confidence:.2f}%</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='text-align: center; color: white; background-color: #4CAF50; padding: 10px; border-radius: 10px;'>"
                        f"<h2>✅ Not Spam ✅</h2>"
                        f"<p>Confidence: {confidence:.2f}%</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )

                # 7. Log the classified message into the database
                log_classification_to_db(input_sms, prediction, confidence)

        except NotFittedError:
            st.error("The model or vectorizer has not been fitted properly. Please check the training process.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown(
    """
    <hr style='border-top: 3px solid #bbb;'>
    <p style='text-align: center; color: grey;'>Built with ❤️ using Streamlit</p>
    """,
    unsafe_allow_html=True
)
