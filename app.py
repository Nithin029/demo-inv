import streamlit as st
import asyncio
import pandas as pd
import os
from image import main as process_pdf
from pdf import create_pdf

st.set_page_config(layout="wide")

# Page state tracking
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'results' not in st.session_state:
    st.session_state.results = None
if 'progress_message' not in st.session_state:
    st.session_state.progress_message = ""
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = ""

# Define display functions for each section
def display_queries_and_answers(queries, query_results):
    st.header("Queries and Answers")

    for i, (query, result) in enumerate(zip(queries, query_results), start=1):
        st.markdown(f"**Query {i}:**")
        st.markdown(f"**Question:** {query}")
        st.markdown(f"**Answer:**")
        st.write(result)

        if i < len(queries):
            st.markdown("---")

def display_other_info(other_info_results):
    st.header("Other Information")
    for category, response in other_info_results.items():
        details_html = f"""
        <details>
            <summary>{category}</summary>
            <div style="margin-left: 20px;">{response}</div>
        </details>
        """
        st.markdown(details_html, unsafe_allow_html=True)

def display_grading_results(grading_results):
    df = {
        "Area/Section": [],
        "Score": [],
        "Weightage": [],
        "Reasoning": []
    }

    gr = grading_results

    for datapoint in gr["sectors"][0]["sections"]:
        df["Area/Section"].append(datapoint["section"])
        df["Score"].append(datapoint["score"])
        df["Weightage"].append(datapoint["weight"])
        df["Reasoning"].append(datapoint["reasoning"])

    grading_df = pd.DataFrame(df)

    st.header(f"Industry: {gr['sectors'][0]['sector']}")
    st.table(data=grading_df)
    st.subheader(f"Estimated score: {gr['final_score']}")

def change_page(step):
    st.session_state.page += step

def streamlit_main():
    st.title("Pitch Deck Analysis")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_name = uploaded_file.name

        # Clear previous results if a new file is uploaded
        st.session_state.results = None
        st.session_state.page = 0

        if st.session_state.results is None:
            progress_bar = st.empty()
            progress_text = st.empty()

            def progress_callback(stage, progress):
                if stage:
                    progress_text.text(stage)
                    progress_bar.progress(progress)
                else:
                    progress_text.empty()
                    progress_bar.empty()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(process_pdf_with_progress(temp_file_path, progress_callback))

            if results is None:
                st.error("File not found or could not be processed.")
                return

            st.session_state.results = results

    if st.session_state.results is not None:
        queries, query_results, other_info_results, grading_results = st.session_state.results

        col1, col2, col3 = st.columns([0.1, 0.8, 0.1])

        with col1:
            if st.session_state.page > 0:
                st.button("◀", key="back_arrow", on_click=lambda: change_page(-1))

        with col2:
            if st.session_state.page == 0:
                display_queries_and_answers(queries, query_results)
            elif st.session_state.page == 1:
                display_other_info(other_info_results)
            elif st.session_state.page == 2:
                display_grading_results(grading_results)

        with col3:
            if st.session_state.page < 2:
                st.button("▶", key="next_arrow", on_click=lambda: change_page(1))

        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] > div:nth-child(1) {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            div[data-testid="stHorizontalBlock"] > div:nth-child(3) {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            div[data-testid="stHorizontalBlock"] > div:nth-child(1) button,
            div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 20px;
                background-color:#90EE90;
                color: black;
            }
            </style>
        """, unsafe_allow_html=True)

        pdf_file = 'output.pdf'
        create_pdf(pdf_file, queries, query_results, other_info_results, grading_results)

        with open(pdf_file, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="output.pdf",
            mime="application/pdf",
        )

async def process_pdf_with_progress(file_path, progress_callback):
    return await process_pdf(file_path, progress_callback)

if __name__ == "__main__":
    streamlit_main()
