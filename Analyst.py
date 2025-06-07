import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.set_page_config(page_title="Groq Excel Analyzer", layout="wide")
st.title("📊 Excel Analyzer + Ask Anything with Groq AI")

uploaded_file = st.file_uploader("📤 Upload Excel file", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("🧾 Basic Info")
        st.write(f"Shape: {df.shape}")
        st.write("Columns:", df.columns.tolist())

        with st.expander("📌 Data Types"):
            st.write(df.dtypes)

        with st.expander("📉 Missing Data Heatmap"):
            plt.figure(figsize=(10, 4))
            sns.heatmap(df.isnull(), cbar=False)
            st.pyplot(plt)

        with st.expander("📈 Summary Statistics"):
            st.write(df.describe(include='all'))

        st.subheader("💬 Ask a question about your data")
        question = st.text_input("Example: What is the average value of column X?")

        if question:
            # Using Groq endpoint via OpenAI-compatible format
            llm = OpenAI(
                api_token="Your key",
                api_base="https://api.groq.com/openai/v1",
                model="llama3-8b-8192"  # You can also use gemma-7b-it or llama3-8b
            )

            smart_df = SmartDataframe(df, config={"llm": llm})

            with st.spinner("Groq is thinking..."):
                answer = smart_df.chat(question)
                st.success("✅ Answer:")
                st.write(answer)

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
else:
    st.info("Upload an Excel (.xlsx) file to begin.")
