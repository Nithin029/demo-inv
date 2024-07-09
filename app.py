import streamlit as st
import asyncio
import pandas as pd
import os
from image import main as process_pdf
from streamlit_option_menu import option_menu

# from image import main as process_pdf
from pdf import create_pdf

st.set_page_config(layout="wide")

# Page state tracking
if "results" not in st.session_state:
    st.session_state.results = None
if "progress_message" not in st.session_state:
    st.session_state.progress_message = ""
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""
if "funding_estimate" not in st.session_state:
    st.session_state.funding_estimate = None


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
            <summary>
            {category}
            </summary>
            <div style="margin-left: 20px;">{response}</div>
        </details>
        """
        st.markdown(details_html, unsafe_allow_html=True)


def display_grading_results(grading_results):
    df = {"Area/Section": [], "Score": [], "Weightage": [], "Reasoning": []}

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


def display_recommendation_results(recommendation_results):
    st.header("Recommendation")
    st.markdown(recommendation_results)


async def process_pdf_with_progress(file_path, funding_estimate, progress_callback):
    return await process_pdf(file_path, funding_estimate, progress_callback)


def streamlit_main():

    with st.sidebar:
        selected = option_menu(
            menu_title="Investment Research",
            options=["FAQ", "Other Grounds", "Scoring", "Recommendation"],
            icons=["patch-question", "graph-up-arrow", "clipboard-check", "star"],
            menu_icon="cash-coin",
        )

    st.title("Pitch Deck Analysis")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.getbuffer())

        st.session_state.uploaded_file_name = uploaded_file.name

        try:
            funding_estimate = st.number_input(
                "Enter your estimate of the funding for the startup (in million USD):",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                value=float('nan')  # This will remove the initial value
            )

            if funding_estimate <= 0:
                st.error("Funding estimate must be greater than 0.")
                return

            st.session_state.funding_estimate = funding_estimate

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
                results = loop.run_until_complete(
                    process_pdf_with_progress(temp_file_path, st.session_state.funding_estimate, progress_callback)
                )

                if results is None:
                    st.error("File not found or could not be processed.")
                    return

                st.session_state.results = results

        except ValueError as e:
            st.error(f"Invalid input: {e}")

    if st.session_state.results is not None:
        queries, query_results, other_info_results, grading_results, recommendation_results = (
            st.session_state.results
        )
        if selected == "FAQ":
            display_queries_and_answers(queries, query_results)
        elif selected == "Other Grounds":
            display_other_info(other_info_results)
        elif selected == "Scoring":
            display_grading_results(grading_results=grading_results)
        elif selected == "Recommendation":
            display_recommendation_results(recommendation_results=recommendation_results)

        st.subheader(f"Funding Estimate: {st.session_state.funding_estimate} million USD")

        pdf_file = "output.pdf"
        create_pdf(
            pdf_file, queries, query_results, other_info_results, grading_results, recommendation_results
        )

        with open(pdf_file, "rb") as pdf_file:
            pdf_data = pdf_file.read()

        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name="output.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    streamlit_main()
