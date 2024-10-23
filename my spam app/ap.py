import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError
from datetime import datetime
import mysql.connector  # Import MySQL connector

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


# Function to connect to MySQL database
def connect_to_db():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Replace with your MySQL username
            password="",  # Replace with your MySQL password
            database="spam_repository"  # Replace with your database name
        )
        return connection
    except mysql.connector.Error as error:
        st.error(f"Failed to connect to MySQL database: {error}")
        return None


# Function to log classification results into MySQL database
def log_to_database(sms_message, prediction, confidence, timestamp):
    connection = connect_to_db()
    if connection:
        try:
            cursor = connection.cursor()
            insert_query = """
                INSERT INTO sms_classification_logs (sms_message, prediction, confidence, classification_time)
                VALUES (%s, %s, %s, %s)
            """
            data = (sms_message, prediction, confidence, timestamp)
            cursor.execute(insert_query, data)
            connection.commit()
            cursor.close()
        except mysql.connector.Error as error:
            st.error(f"Failed to log to MySQL database: {error}")
        finally:
            connection.close()


# Function to log errors to MySQL database
def log_error_to_db(error_message):
    connection = connect_to_db()
    if connection:
        try:
            cursor = connection.cursor()
            insert_query = """
                INSERT INTO error_logs (error_message, error_time)
                VALUES (%s, %s)
            """
            data = (error_message, datetime.now())
            cursor.execute(insert_query, data)
            connection.commit()
            cursor.close()
        except mysql.connector.Error as error:
            st.error(f"Failed to log error: {error}")
        finally:
            connection.close()


# Function to display classification logs from MySQL database
def display_classification_logs():
    connection = connect_to_db()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT sms_message, prediction, confidence, classification_time 
                FROM sms_classification_logs 
                ORDER BY classification_time DESC 
                LIMIT 5
            """)
            logs = cursor.fetchall()
            cursor.close()
            return logs
        except mysql.connector.Error as error:
            st.error(f"Failed to retrieve logs from MySQL database: {error}")
            return []
        finally:
            connection.close()
    return []


# Function to display spam count
def display_spam_count():
    connection = connect_to_db()
    if connection:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM sms_classification_logs WHERE prediction = 'Spam'
            """)
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except mysql.connector.Error as error:
            st.error(f"Failed to retrieve spam count: {error}")
            return 0
        finally:
            connection.close()
    return 0


# Load the vectorizer and model separately
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("The required files (vectorizer.pkl or model.pkl) were not found.")
except Exception as e:
    st.error(f"An error occurred while loading the files: {e}")
    log_error_to_db(f"Model/Vectorizer loading error: {e}")

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
st.markdown('<div class="main-header">📱 SMS Spam Filtering System</div>', unsafe_allow_html=True)
st.write("<p style='text-align: center;'>Detect whether an SMS message is <strong>Spam</strong> or <strong>Not Spam</strong>.</p>", unsafe_allow_html=True)

# Display spam count
spam_count = display_spam_count()
st.markdown(f"### Spam Messages Detected So Far: {spam_count}")

# Input text box for the user with a customized style
input_sms = st.text_area(
    "Enter the message below:",
    height=150,
    placeholder="Type your SMS message here...",
    key="input",
    help="Type the SMS message you want to classify."
)

# Predict button with custom style
if st.button('Predict 🚀'):
    if input_sms.strip() == "":
        st.warning("Please enter an SMS message to classify.")
    else:
        try:
            # Preprocess the input text
            transformed_sms = transform_text(input_sms)

            # Vectorize the transformed text
            vectorized_sms = vectorizer.transform([transformed_sms])

            # Make the prediction using the loaded model and get confidence score
            prediction = model.predict(vectorized_sms)[0]
            confidence = model.predict_proba(vectorized_sms)[0][prediction] * 100  # Confidence percentage

            # Display the result with enhanced visuals and confidence score
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

            # Log the classified message to the MySQL database
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_to_database(input_sms, 'Spam' if prediction == 1 else 'Not Spam', confidence, timestamp)

        except NotFittedError:
            st.error("The model or vectorizer has not been fitted properly. Please check the training process.")
            log_error_to_db("Model not fitted error.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            log_error_to_db(f"Prediction error: {e}")

# Display the last few classified messages
st.markdown("### Recently Classified Messages")
logs = display_classification_logs()
if logs:
    for log in logs:
        st.write(f"**Message:** {log[0]} | **Prediction:** {log[1]} | **Confidence:** {log[2]:.2f}% | **Time:** {log[3]}")
else:
    st.write("No classification logs available.")

# Footer
st.markdown(
    """
    <hr style='border-top: 3px solid #bbb;'>
    <p style='text-align: center; color: grey;'>My Final Year Project❤️ using Streamlit</p>
    """,
    unsafe_allow_html=True
)
