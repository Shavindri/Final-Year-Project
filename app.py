import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import streamlit.components.v1 as components

from io import BytesIO
from PIL import Image, ImageDraw
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Online Banking Cybersecurity Awareness Toolkit",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("Online_Banking_Cleaned_Dataset (1).csv")

df = load_data()

st.sidebar.title("Navigation")

pages = [
    "Home",
    "Dataset Overview",
    "Exploratory Data Analysis",
    "Cyber Awareness Assessment",
    "Biometric Training Demo",
    "Learning Resources"
]

page = st.sidebar.radio("Go to", pages)

if page == "Home":

    st.title("Online Banking Cybersecurity Awareness Toolkit")

    st.markdown("""
    ### Evaluating the Effectiveness of User Awareness and Front-End Security Measures in Online Banking
    """)

    st.write("---")

    col1, col2 = st.columns([2,1])

    with col1:

        st.write("""
        Welcome to the Online Banking Cybersecurity Awareness Toolkit.

        This application was developed as part of a Final Year Research Project to
        evaluate users' cybersecurity awareness and the effectiveness of front-end
        security measures used in online banking.

        The toolkit combines survey findings, exploratory data analysis,
        cybersecurity awareness assessment, personalised recommendations,
        and an educational biometric authentication demonstration.
        """)

    with col2:
        st.metric("Survey Responses", len(df))
        st.metric("Questionnaire Items", len(df.columns))
        st.metric("Toolkit Modules", len(pages))

    st.write("---")

    st.subheader("Toolkit Features")

    col1, col2 = st.columns(2)

    with col1:

        st.success("Cyber Awareness Assessment")

        st.write("""
        • Assess users' cybersecurity awareness

        • Calculate awareness percentage

        • Provide personalised recommendations
        """)

        st.success("Exploratory Data Analysis")

        st.write("""
        • Interactive charts

        • Dataset overview

        • Correlation analysis
        """)

    with col2:

        st.success("Biometric Training Demonstration")

        st.write("""
        • Face detection explanation

        • Facial landmark identification

        • Faceprint generation concept
        """)

        st.success("Learning Resources")

        st.write("""
        • Phishing awareness

        • Password security

        • Multi-Factor Authentication

        • AI-enabled cyber threats
        """)

    st.write("---")

    with st.expander("Project Aim"):

        st.write("""
        To evaluate the effectiveness of user awareness and front-end security
        measures in online banking and to develop an interactive educational
        toolkit that improves users' cybersecurity knowledge and promotes safer
        online banking practices.
        """)

    with st.expander("How to Use This Toolkit"):

        st.write("""
        1. Review the dataset overview.

        2. Explore the visualisations.

        3. Complete the Cyber Awareness Assessment.

        4. Read your personalised recommendations.

        5. Learn how biometric authentication works through the training demonstration.
        """)


elif page == "Dataset Overview":

    st.title("Dataset Overview")

    st.write("""
    This dataset was collected through an online questionnaire designed to
    evaluate users' cybersecurity awareness and their perceptions of front-end
    security measures in online banking.

    It includes demographic information, online banking usage, cybersecurity
    awareness, phishing experiences, practical security scenarios, security
    behaviour, opinions about authentication controls and biometric awareness.
    """)

    st.write("---")

    
    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Responses", df.shape[0])
    col2.metric("Total Variables", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Records", int(df.duplicated().sum()))

    st.write("---")

    

    st.subheader("Dataset Information")

    dataset_info = pd.DataFrame({
        "Property": [
            "Number of Rows",
            "Number of Columns",
            "Missing Values",
            "Duplicate Records",
            "Memory Usage"
        ],
        "Value": [
            df.shape[0],
            df.shape[1],
            int(df.isnull().sum().sum()),
            int(df.duplicated().sum()),
            f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    st.subheader("Survey Structure")

    survey_structure = pd.DataFrame({
        "Survey Section": [
            "Demographic Information",
            "Cybersecurity Awareness",
            "Cybersecurity Experience",
            "Practical Security Scenarios",
            "Opinions on Security Measures",
            "Online Banking Behaviour",
            "Biometric Authentication",
            "Other Variables"
        ],
        "Number of Variables": [
            7,
            7,
            6,
            4,
            5,
            4,
            6,
            2
        ]
    })

    st.dataframe(
        survey_structure,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    st.subheader("Dataset Preview")

    preview_rows = st.slider(
        "Select the number of rows to display",
        min_value=5,
        max_value=min(20, len(df)),
        value=5
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True
    )

    st.write("---")

    st.subheader("Data Types")

    datatype_table = pd.DataFrame({
        "Variable": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        datatype_table,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")


    st.subheader("Data Type Summary")

    datatype_summary = (
        df.dtypes.astype(str)
        .value_counts()
        .reset_index()
    )

    datatype_summary.columns = [
        "Data Type",
        "Number of Variables"
    ]

    st.dataframe(
        datatype_summary,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")


    st.subheader("Missing Value Analysis")

    missing_table = pd.DataFrame({
        "Variable": df.columns,
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage": (
            df.isnull().mean().values * 100
        ).round(2)
    })

    st.dataframe(
        missing_table,
        use_container_width=True,
        hide_index=True
    )

    if df.isnull().sum().sum() == 0:
        st.success(
            "The cleaned dataset contains no missing values."
        )

    st.write("---")

    st.subheader("Numerical Variable Summary")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numerical_columns) > 0:

        numerical_summary = (
            df[numerical_columns]
            .describe()
            .transpose()
            .reset_index()
        )

        numerical_summary = numerical_summary.rename(
            columns={"index": "Variable"}
        )

        st.dataframe(
            numerical_summary,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No numerical variables are available."
        )

    st.write("---")


    st.subheader("Categorical Variable Summary")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_columns) > 0:

        categorical_summary = (
            df[categorical_columns]
            .describe()
            .transpose()
            .reset_index()
        )

        categorical_summary = categorical_summary.rename(
            columns={"index": "Variable"}
        )

        st.dataframe(
            categorical_summary,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No categorical variables are available."
        )


elif page == "Exploratory Data Analysis":

    st.title("Exploratory Data Analysis")

    st.write("""
    This page presents univariate, bivariate and multivariate analyses of the
    survey dataset. Use the filters to explore how demographic characteristics,
    online banking usage and cybersecurity-related scores vary across respondents.
    """)

    st.subheader("Interactive Filters")

    col1, col2, col3 = st.columns(3)

    with col1:

        age_options = df["Age"].dropna().unique().tolist()

        age_filter = st.multiselect(
            "Age Group",
            options=age_options,
            default=age_options
        )

    with col2:

        gender_options = df["Gender"].dropna().unique().tolist()

        gender_filter = st.multiselect(
            "Gender",
            options=gender_options,
            default=gender_options
        )

    with col3:

        banking_options = (
            df["Banking_Use"]
            .dropna()
            .unique()
            .tolist()
        )

        banking_filter = st.multiselect(
            "Banking Usage",
            options=banking_options,
            default=banking_options
        )

    filtered_df = df[
        df["Age"].isin(age_filter)
        & df["Gender"].isin(gender_filter)
        & df["Banking_Use"].isin(banking_filter)
    ].copy()

    if filtered_df.empty:

        st.warning(
            "No records match the selected filters. "
            "Please change the selected options."
        )

    else:

        st.write("---")
        st.subheader("Filtered Dataset Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Responses",
            len(filtered_df)
        )

        c2.metric(
            "Average Awareness Score",
            f"{filtered_df['Awareness_Score'].mean():.2f}"
        )

        c3.metric(
            "Average Behaviour Score",
            f"{filtered_df['Behaviour_Score'].mean():.2f}"
        )

        c4.metric(
            "Average Biometric Score",
            f"{filtered_df['Biometric_Score'].mean():.2f}"
        )


        st.write("---")
        st.header("Univariate Analysis")

        st.write("""
        Univariate analysis examines one variable at a time. It is used to
        understand the frequency, distribution, central tendency and spread
        of individual variables.
        """)


        st.subheader("Categorical Variable Distributions")

        categorical_variable = st.selectbox(
            "Select a categorical variable",
            [
                "Age",
                "Gender",
                "Education",
                "Banking_Use",
                "Platform",
                "Awareness_Level",
                "Behaviour_Level",
                "Biometric_Level",
                "Biggest_Concern"
            ]
        )

        category_counts = (
            filtered_df[categorical_variable]
            .value_counts()
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.countplot(
                data=filtered_df,
                x=categorical_variable,
                order=category_counts.index,
                ax=ax
            )

            ax.set_title(
                f"Distribution of {categorical_variable.replace('_', ' ')}"
            )

            ax.set_xlabel(
                categorical_variable.replace("_", " ")
            )

            ax.set_ylabel("Number of Respondents")

            plt.xticks(
                rotation=35,
                ha="right"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.pie(
                category_counts.values,
                labels=category_counts.index,
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title(
                f"Percentage Distribution of "
                f"{categorical_variable.replace('_', ' ')}"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        st.subheader("Score Distribution")

        score_columns = [
            "Awareness_Score",
            "Opinion_Score",
            "Behaviour_Score",
            "Biometric_Score"
        ]

        score_variable = st.selectbox(
            "Select a score variable",
            score_columns,
            key="score_histogram_variable"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig, ax = plt.subplots(figsize=(8, 5))

            sns.histplot(
                filtered_df[score_variable].dropna(),
                bins=8,
                kde=True,
                ax=ax
            )

            ax.set_title(
                f"Histogram of {score_variable.replace('_', ' ')}"
            )

            ax.set_xlabel(
                score_variable.replace("_", " ")
            )

            ax.set_ylabel("Frequency")

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

        with col2:

            fig, ax = plt.subplots(figsize=(7, 5))

            sns.boxplot(
                y=filtered_df[score_variable],
                ax=ax
            )

            ax.set_title(
                f"Box Plot of {score_variable.replace('_', ' ')}"
            )

            ax.set_ylabel(
                score_variable.replace("_", " ")
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)


        st.subheader("Descriptive Statistics")

        descriptive_table = (
            filtered_df[score_columns]
            .describe()
            .transpose()
            .reset_index()
        )

        descriptive_table = descriptive_table.rename(
            columns={"index": "Score Variable"}
        )

        st.dataframe(
            descriptive_table,
            use_container_width=True,
            hide_index=True
        )


        st.write("---")
        st.header("Bivariate Analysis")

        st.write("""Bivariate analysis examines the relationship between two variables.
        The charts below compare cybersecurity scores with demographic and
        online banking characteristics.
        """)

        st.subheader("Group Comparison")

        col1, col2 = st.columns(2)

        with col1:

            grouping_variable = st.selectbox(
                "Select a grouping variable",
                [
                    "Age",
                    "Gender",
                    "Education",
                    "Banking_Use",
                    "Platform"
                ]
            )

        with col2:

            comparison_score = st.selectbox(
                "Select a score to compare",
                score_columns,
                key="group_score"
            )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(
            data=filtered_df,
            x=grouping_variable,
            y=comparison_score,
            ax=ax
        )

        ax.set_title(
            f"{comparison_score.replace('_', ' ')} by "
            f"{grouping_variable.replace('_', ' ')}"
        )

        ax.set_xlabel(
            grouping_variable.replace("_", " ")
        )

        ax.set_ylabel(
            comparison_score.replace("_", " ")
        )

        plt.xticks(
            rotation=35,
            ha="right"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Relationship Between Scores")

        col1, col2 = st.columns(2)

        with col1:

            x_score = st.selectbox(
                "Select X-axis score",
                score_columns,
                index=0
            )

        with col2:

            y_score = st.selectbox(
                "Select Y-axis score",
                score_columns,
                index=2
            )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.regplot(
            data=filtered_df,
            x=x_score,
            y=y_score,
            ax=ax
        )

        ax.set_title(
            f"Relationship Between "
            f"{x_score.replace('_', ' ')} and "
            f"{y_score.replace('_', ' ')}"
        )

        ax.set_xlabel(
            x_score.replace("_", " ")
        )

        ax.set_ylabel(
            y_score.replace("_", " ")
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        correlation_value = (
            filtered_df[[x_score, y_score]]
            .corr()
            .iloc[0, 1]
        )

        st.metric(
            "Pearson Correlation",
            f"{correlation_value:.3f}"
        )

        st.subheader("Awareness Level by Gender")

        stacked_table = pd.crosstab(
            filtered_df["Gender"],
            filtered_df["Awareness_Level"]
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        stacked_table.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        ax.set_title(
            "Awareness Level by Gender"
        )

        ax.set_xlabel("Gender")
        ax.set_ylabel("Number of Respondents")

        plt.xticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write("---")
        st.header("Multivariate Analysis")

        st.write("""
        Multivariate analysis examines relationships among three or more
        variables. It helps identify broader patterns across awareness,
        behaviour, opinions and biometric understanding.
        """)

        st.subheader("Correlation Heatmap")

        correlation_matrix = (
            filtered_df[score_columns]
            .corr()
        )

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            square=True,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Correlation Between Composite Scores"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Pairwise Score Analysis")

        pairplot_data = (
            filtered_df[score_columns]
            .dropna()
        )

        if len(pairplot_data) > 1:

            pair_grid = sns.pairplot(
                pairplot_data,
                diag_kind="hist"
            )

            pair_grid.fig.suptitle(
                "Pairwise Relationships Between Composite Scores",
                y=1.02
            )

            st.pyplot(pair_grid.fig)

            plt.close(pair_grid.fig)

        else:

            st.info(
                "Not enough records are available for pairwise analysis."
            )
        st.subheader("Bubble Chart")

        fig, ax = plt.subplots(figsize=(9, 6))

        bubble_sizes = (
            filtered_df["Opinion_Score"]
            .fillna(0)
            .clip(lower=0)
            * 100
        )

        scatter = ax.scatter(
            filtered_df["Awareness_Score"],
            filtered_df["Behaviour_Score"],
            s=bubble_sizes,
            alpha=0.6
        )

        ax.set_title(
            "Awareness, Behaviour and Opinion Scores"
        )

        ax.set_xlabel("Awareness Score")
        ax.set_ylabel("Behaviour Score")

        st.caption(
            "The position represents awareness and behaviour scores. "
            "Bubble size represents the opinion score."
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.subheader("Average Composite Scores")

        average_scores = (
            filtered_df[score_columns]
            .mean()
            .reset_index()
        )

        average_scores.columns = [
            "Score Type",
            "Average Score"
        ]

        average_scores["Score Type"] = (
            average_scores["Score Type"]
            .str.replace("_Score", "", regex=False)
        )

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.barplot(
            data=average_scores,
            x="Score Type",
            y="Average Score",
            ax=ax
        )

        ax.set_title(
            "Average Composite Score Comparison"
        )

        ax.set_xlabel("Composite Measure")
        ax.set_ylabel("Average Score")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.write("---")
        st.header("Key Findings")

        awareness_mean = (
            filtered_df["Awareness_Score"]
            .mean()
        )

        behaviour_mean = (
            filtered_df["Behaviour_Score"]
            .mean()
        )

        biometric_mean = (
            filtered_df["Biometric_Score"]
            .mean()
        )

        awareness_biometric_corr = (
            filtered_df[
                [
                    "Awareness_Score",
                    "Biometric_Score"
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        st.write(
            f"""The selected dataset contains **{len(filtered_df)} respondents**.
            The average awareness score is **{awareness_mean:.2f}**, while the
            average behaviour score is **{behaviour_mean:.2f}** and the average
            biometric score is **{biometric_mean:.2f}**.

            The correlation between awareness and biometric understanding is
            **{awareness_biometric_corr:.3f}**. The visualisations show how
            cybersecurity awareness and behaviour vary across demographic groups
            and online banking usage patterns.
            """
        )

        st.write("---")
        st.subheader("Filtered Dataset")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        filtered_csv = (
            filtered_df
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="Download Filtered Dataset",
            data=filtered_csv,
            file_name="Filtered_Data.csv",
            mime="text/csv"
        )


elif page == "Cyber Awareness Assessment":
    st.title("Cyber Awareness Assessment")

    st.write("""
    Answer the following questions. The application will calculate your cyber awareness score
    and provide personalised feedback.
    """)

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
        feedback.append("Never share OTPs. Banks will not ask customers to share OTPs, passwords, or PINs.")

    password = st.selectbox(
        "3. How often do you use a unique password for online banking?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if password == "Always":
        score += 1
    else:
        feedback.append("Improve password practices by using a strong and unique password for online banking.")

    mfa = st.selectbox(
        "4. How often do you enable Multi-Factor Authentication when it is available?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if mfa == "Always":
        score += 1
    else:
        feedback.append("Enable Multi-Factor Authentication to add an extra layer of protection.")

    website = st.selectbox(
        "5. How often do you verify the banking website before entering login details?",
        ["Always", "Sometimes", "Rarely", "Never"]
    )

    if website == "Always":
        score += 1
    else:
        feedback.append("Always check website authenticity before entering banking credentials.")

    public_wifi = st.selectbox(
        "6. Would you use public Wi-Fi for online banking?",
        ["Never", "Sometimes", "Often"]
    )

    if public_wifi == "Never":
        score += 1
    else:
        feedback.append("Avoid public Wi-Fi for online banking because it may expose sensitive information.")

    biometric = st.selectbox(
        "7. Do you understand how biometric authentication protects online banking accounts?",
        ["Yes", "Somewhat", "No"]
    )

    if biometric == "Yes":
        score += 1
    else:
        feedback.append("Improve understanding of biometric authentication and how biometric data is protected.")

    ai_scams = st.selectbox(
        "8. What would you do if you received a voice call that sounded like your bank manager asking you to approve a transaction?",
        [
            "Verify the request using the bank's official contact details",
            "End the call without checking",
            "Approve the transaction"
        ]
    )

    if ai_scams == "Verify the request using the bank's official contact details":
        score += 1
    else:
        feedback.append("Be cautious of AI-generated scams and deepfake calls. Always verify urgent requests officially.")

    if st.button("Calculate Awareness Score"):
        percentage = round((score / total) * 100, 2)

        st.subheader("Your Cyber Awareness Score")
        st.write(f"{percentage}%")

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

    st.write("""
    This section explains how biometric facial identification works in a simplified way.
    It is not a real authentication system and does not store or verify biometric data.
    """)

    uploaded_image = st.file_uploader(
        "Upload a face image for demonstration purposes",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        st.subheader("Step 1: Face Detection")
        st.write("""
        A biometric system first detects whether a face is present in the image.
        """)

        st.subheader("Step 2: Facial Landmark Identification")
        st.write("""
        The system identifies key facial landmarks such as the eyes, nose, mouth, jawline,
        and distance between facial features.
        """)

        st.subheader("Step 3: Faceprint Generation")
        st.write("""
        These facial measurements are converted into a mathematical representation known as a faceprint.
        A faceprint is not the same as a normal image; it is a numerical pattern used for comparison.
        """)

        st.subheader("Step 4: Matching Process")
        st.write("""
        In a real authentication system, the generated faceprint would be compared with a stored template
        to decide whether access should be granted.
        """)

        st.warning("""
        This demonstration is educational only. It does not authenticate users, store biometric data,
        or perform real facial recognition.
        """)

        st.subheader("Privacy and Security Considerations")
        st.write("""
        Biometric systems must protect user privacy by securely storing biometric templates,
        limiting access to sensitive data, and clearly explaining how biometric information is used.
        """)


elif page == "Learning Resources":
    st.title("Learning Resources")

    topic = st.selectbox(
        "Choose a topic",
        [
            "Phishing Awareness",
            "OTP Security",
            "Password Security",
            "Multi-Factor Authentication",
            "Safe Online Banking Behaviour",
            "Biometric Authentication",
            "AI Scams and Deepfake Fraud"
        ]
    )

    if topic == "Phishing Awareness":
        st.write("""
        Phishing is a method used by cybercriminals to trick users into revealing sensitive information.
        Users should avoid clicking suspicious links and should verify messages through official bank channels.
        """)

    elif topic == "OTP Security":
        st.write("""
        OTPs should never be shared with anyone. Legitimate banks do not ask customers to reveal OTPs,
        passwords, or PINs.
        """)

    elif topic == "Password Security":
        st.write("""
        Online banking users should use strong, unique passwords and avoid reusing passwords across accounts.
        """)

    elif topic == "Multi-Factor Authentication":
        st.write("""
        Multi-Factor Authentication strengthens account security by requiring more than one method of verification.
        """)

    elif topic == "Safe Online Banking Behaviour":
        st.write("""
        Safe online banking behaviour includes avoiding public Wi-Fi, updating banking apps,
        verifying website authenticity, and monitoring security alerts.
        """)

    elif topic == "Biometric Authentication":
        st.write("""
        Biometric authentication uses unique physical characteristics such as fingerprints or facial recognition.
        Users should understand how biometric data is stored, protected, and used.
        """)

    elif topic == "AI Scams and Deepfake Fraud":
        st.write("""
        AI-generated scams and deepfake fraud can imitate real voices, images, or messages.
        Users should verify urgent financial requests through official banking channels.
        """)
