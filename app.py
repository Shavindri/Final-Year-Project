import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(
    page_title="Online Banking Cybersecurity Awareness Toolkit",
    layout="wide"
)

DATA_PATH = "Online_Banking_Cleaned_Dataset.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("Dataset not found. Make sure Online_Banking_Cleaned_Dataset.csv exists in the repository.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Data Overview",
        "EDA",
        "Cyber Awareness Prediction",
        "Biometric Training Demo"
    ]
)

if page == "Home":
    st.title("Online Banking Cybersecurity Awareness Toolkit")

    st.write(
        "This application provides an interactive cybersecurity awareness toolkit "
        "for online banking users. It analyses survey findings, estimates users' "
        "cyber awareness level, provides personalised feedback, and includes a "
        "training demonstration explaining biometric facial identification."
    )


elif page == "Data Overview":
    st.title("Dataset Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.subheader("Dataset Shape")
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    st.subheader("Summary Statistics")
    st.dataframe(df.describe(include="all").T)

    st.subheader("Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df)


elif page == "EDA":
    st.title("Exploratory Data Analysis Dashboard")

    st.subheader("Demographic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        df["Age"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Number of Respondents")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df["Gender"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Gender Distribution")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Respondents")
        st.pyplot(fig)

    st.subheader("Online Banking Usage")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        df["Banking_Use"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Online Banking Usage Frequency")
        ax.set_xlabel("Usage Frequency")
        ax.set_ylabel("Number of Respondents")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        df["Platform"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Online Banking Platform")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Number of Respondents")
        st.pyplot(fig)

    st.subheader("Awareness, Behaviour and Biometric Levels")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(4, 3))
        df["Awareness_Level"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Awareness Level")
        ax.set_ylabel("Respondents")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        df["Behaviour_Level"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Behaviour Level")
        ax.set_ylabel("Respondents")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(4, 3))
        df["Biometric_Level"].value_counts().plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_title("Biometric Understanding Level")
        ax.set_ylabel("Respondents")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    corr_cols = [
        "Awareness_Score",
        "Opinion_Score",
        "Behaviour_Score",
        "Biometric_Score"
    ]

    corr_cols = [c for c in corr_cols if c in df.columns]

    if len(corr_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Correlation columns are not available in the dataset.")


elif page == "Cyber Awareness Prediction":
    st.title("Cyber Awareness Prediction")

    st.write(
        "Answer the following questions. The application will estimate your cyber awareness "
        "level and provide personalised feedback."
    )

    score = 0
    total = 8
    feedback = []

    phishing = st.selectbox(
        "1. What would you do if you received an email claiming your bank account will be suspended unless you click a link?",
        [
            "Verify the request using the bank's official website or customer service",
            "Ignore it completely",
            "Click the link immediately"
        ]
    )

    if phishing == "Verify the request using the bank's official website or customer service":
        score += 1
    else:
        feedback.append("Improve phishing awareness by verifying suspicious emails through official bank channels.")

    otp = st.selectbox(
        "2. What would you do if someone claiming to be from your bank asks for your OTP?",
        [
            "End the call and contact the bank through official channels",
            "Ask why it is needed",
            "Ignore the request but continue the conversation",
            "Share the OTP"
        ]
    )

    if otp == "End the call and contact the bank through official channels":
        score += 1
    else:
        feedback.append("Never share OTPs, passwords, or PINs with anyone.")

    password = st.selectbox(
        "3. How often do you use a unique password for online banking?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if password == "Always":
        score += 1
    else:
        feedback.append("Improve password practices by using a strong and unique online banking password.")

    mfa = st.selectbox(
        "4. How often do you enable Multi-Factor Authentication when it is available?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if mfa == "Always":
        score += 1
    else:
        feedback.append("Enable Multi-Factor Authentication whenever it is available.")

    website = st.selectbox(
        "5. How often do you verify the banking website before entering login details?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if website == "Always":
        score += 1
    else:
        feedback.append("Always verify website authenticity before entering banking credentials.")

    public_wifi = st.selectbox(
        "6. Would you use public Wi-Fi for online banking?",
        ["Never", "Sometimes", "Often"]
    )

    if public_wifi == "Never":
        score += 1
    else:
        feedback.append("Avoid public Wi-Fi when accessing online banking services.")

    biometric = st.selectbox(
        "7. Do you understand how biometric authentication protects online banking accounts?",
        ["Yes", "Somewhat", "No"]
    )

    if biometric == "Yes":
        score += 1
    else:
        feedback.append("Improve your understanding of biometric authentication and biometric data protection.")

    deepfake = st.selectbox(
        "8. What would you do if a voice call sounded like your bank manager and asked you to approve a transaction?",
        [
            "Verify the request using the bank's official contact details",
            "End the call without checking",
            "Approve the transaction"
        ]
    )

    if deepfake == "Verify the request using the bank's official contact details":
        score += 1
    else:
        feedback.append("Be cautious of AI-generated scams and deepfake calls. Always verify urgent requests officially.")

    if st.button("Calculate Awareness Score"):
        percentage = round((score / total) * 100, 2)

        st.subheader("Cyber Awareness Score")
        st.write(f"{percentage}%")

        st.progress(int(percentage))

        if percentage >= 75:
            st.success("Awareness Level: High")
        elif percentage >= 50:
            st.warning("Awareness Level: Moderate")
        else:
            st.error("Awareness Level: Low")

        st.subheader("Personalised Feedback")

        if feedback:
            for item in feedback:
                st.write("- " + item)
        else:
            st.write("You demonstrated strong cyber awareness across all assessed areas.")


elif page == "Biometric Training Demo":
    st.title("Biometric Facial Identification Training Demonstration")

    st.write(
        "This section provides a simple web-based training demonstration to explain "
        "how biometric facial identification works. It is not a real authentication system."
    )

    uploaded_image = st.file_uploader(
        "Upload a face image for educational demonstration",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        st.subheader("Step 1: Face Detection")
        st.write("The system first checks whether a face is present in the uploaded image.")

        st.subheader("Step 2: Facial Landmark Detection")
        st.write(
            "A real system identifies key facial landmarks such as the eyes, nose, mouth, "
            "jawline, and the distance between facial features."
        )

        st.subheader("Step 3: Faceprint Generation")
        st.write(
            "These facial features are converted into a mathematical representation known as a faceprint. "
            "A faceprint is a numerical pattern rather than a normal photograph."
        )

        st.subheader("Step 4: Matching")
        st.write(
            "In a real authentication system, the generated faceprint would be compared with a stored "
            "template to decide whether access should be granted."
        )

        st.subheader("Privacy and Security")
        st.write(
            "Biometric data must be stored securely and protected using strong data protection practices. "
            "Users should understand how their biometric data is collected, stored, and used."
        )

        st.warning(
            "This demonstration is for educational purposes only. It does not authenticate users or store biometric data."
        )
