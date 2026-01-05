import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="StudyTrack AI", layout="wide")

# -------------------------------
# SESSION STATE
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------
# LOGIN PAGE
# -------------------------------
def login_page():
    st.title("🔐 Login – StudyTrack AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# -------------------------------
# LOGOUT
# -------------------------------
def logout():
    st.session_state.logged_in = False
    st.rerun()

# -------------------------------
# FILE READER
# -------------------------------
def read_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        return None

# -------------------------------
# ATTENTION LEVEL (RULE BASED)
# -------------------------------
def get_attention_level(study, sleep, social):
    if study >= 6 and sleep >= 7 and social <= 2:
        return "High"
    elif study >= 4 and sleep >= 6:
        return "Medium"
    else:
        return "Low"

# -------------------------------
# RECOMMENDATION LOGIC
# -------------------------------
def generate_recommendation(marks):
    if marks >= 85:
        return "Excellent performance. Maintain current study habits."
    elif marks >= 70:
        return "Good performance. Focus on weak subjects."
    elif marks >= 50:
        return "Average performance. Increase study hours and reduce distractions."
    else:
        return "Needs improvement. Follow a structured study plan."

# -------------------------------
# DASHBOARD
# -------------------------------
def dashboard():
    st.sidebar.title("📊 Study Tracker")
    menu = st.sidebar.radio(
        "Navigation",
        ["Home", "Train Model", "Batch Prediction", "Insights", "Logout"]
    )

    # ---------------- HOME ----------------
    if menu == "Home":
        st.title("🎓 StudyTrack – AI based Student Study Habit Recommender")
        st.write("""
        This system predicts student marks, infers attention levels,
        and provides personalized study habit recommendations using AI.
        """)

    # ---------------- TRAIN MODEL ----------------
    elif menu == "Train Model":
        st.header("🧠 Train Marks Prediction Model")

        train_file = st.file_uploader(
            "Upload Training Dataset (CSV or Excel)",
            type=["csv", "xlsx"]
        )

        if train_file:
            df = read_file(train_file)
            st.dataframe(df)

            features = ["StudyHours", "SleepHours", "SocialMedia", "Exercise"]
            target = "Marks"

            X = df[features]
            y = df[target]

            model = LinearRegression()
            model.fit(X, y)

            st.session_state.marks_model = model
            st.success("Marks prediction model trained successfully")

    # ---------------- BATCH PREDICTION ----------------
    elif menu == "Batch Prediction":
        st.header("📂 Batch Prediction")

        if "marks_model" not in st.session_state:
            st.warning("Please train the model first")
            return

        pred_file = st.file_uploader(
            "Upload Prediction Dataset (CSV or Excel)",
            type=["csv", "xlsx"]
        )

        if pred_file:
            df = read_file(pred_file)
            st.dataframe(df)

            features = ["StudyHours", "SleepHours", "SocialMedia", "Exercise"]
            X = df[features]

            df["Predicted_Marks"] = st.session_state.marks_model.predict(X)

            df["Attention_Level"] = df.apply(
                lambda x: get_attention_level(
                    x["StudyHours"],
                    x["SleepHours"],
                    x["SocialMedia"]
                ),
                axis=1
            )

            df["Recommendation"] = df["Predicted_Marks"].apply(generate_recommendation)

            st.success("Batch prediction completed")
            st.dataframe(df)

            st.download_button(
                "⬇️ Download Results",
                df.to_csv(index=False),
                "prediction_results.csv",
                "text/csv"
            )

            st.session_state.predicted_df = df

    # ---------------- INSIGHTS ----------------
    elif menu == "Insights":
        st.header("📈 Insights & Visualizations")

        if "predicted_df" not in st.session_state:
            st.warning("Run batch prediction first")
            return

        df = st.session_state.predicted_df

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Marks", round(df["Predicted_Marks"].mean(), 2))
        col2.metric("High Attention (%)", round((df["Attention_Level"] == "High").mean() * 100, 2))
        col3.metric("Low Attention (%)", round((df["Attention_Level"] == "Low").mean() * 100, 2))

        st.subheader("🏆 Performance Extremes")

        top = df.nlargest(5, "Predicted_Marks")[["Student_Name", "Predicted_Marks"]]
        bottom = df.nsmallest(5, "Predicted_Marks")[["Student_Name", "Predicted_Marks"]]

        c1, c2 = st.columns(2)
        c1.dataframe(top)
        c2.dataframe(bottom)

        st.subheader("📊 Feature Correlation")
        fig, ax = plt.subplots()
        sns.heatmap(
            df[["StudyHours", "SleepHours", "SocialMedia", "Exercise", "Predicted_Marks"]].corr(),
            annot=True, cmap="viridis", ax=ax
        )
        st.pyplot(fig)

        st.subheader("📉 Distribution of Predicted Marks")

        fig1, ax1 = plt.subplots()
        ax1.hist(df["Predicted_Marks"], bins=10)
        ax1.set_xlabel("Predicted Marks")
        ax1.set_ylabel("Number of Students")
        st.pyplot(fig1)

        st.subheader("🧠 Attention Level Distribution")

        attention_counts = df["Attention_Level"].value_counts()

        fig2, ax2 = plt.subplots()
        ax2.bar(attention_counts.index, attention_counts.values)
        ax2.set_xlabel("Attention Level")
        ax2.set_ylabel("Number of Students")
        st.pyplot(fig2)

        st.subheader("📚 Study Hours vs Predicted Marks")

        fig3, ax3 = plt.subplots()
        ax3.scatter(df["StudyHours"], df["Predicted_Marks"])
        ax3.set_xlabel("Study Hours")
        ax3.set_ylabel("Predicted Marks")
        st.pyplot(fig3)


        st.subheader("😴 Sleep Hours vs Predicted Marks")

        fig4, ax4 = plt.subplots()
        ax4.scatter(df["SleepHours"], df["Predicted_Marks"])
        ax4.set_xlabel("Sleep Hours")
        ax4.set_ylabel("Predicted Marks")
        st.pyplot(fig4)


        st.subheader("📱 Social Media Usage vs Predicted Marks")

        fig5, ax5 = plt.subplots()
        ax5.scatter(df["SocialMedia"], df["Predicted_Marks"])
        ax5.set_xlabel("Social Media Hours")
        ax5.set_ylabel("Predicted Marks")
        st.pyplot(fig5)

        st.subheader("🎯 Recommendation Distribution")

        rec_counts = df["Recommendation"].value_counts()

        fig6, ax6 = plt.subplots()
        ax6.barh(rec_counts.index, rec_counts.values)
        ax6.set_xlabel("Number of Students")
        st.pyplot(fig6)



    # ---------------- LOGOUT ----------------
    elif menu == "Logout":
        logout()

# -------------------------------
# APP START
# -------------------------------
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()
