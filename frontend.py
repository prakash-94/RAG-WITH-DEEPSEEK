import streamlit as st

from rag_pipeline import answer_query, llm_model, retrieve_docs
from vector_database import get_or_create_vector_store


st.set_page_config(page_title="PDF RAG Assistant", page_icon=":books:")
st.title("PDF RAG Assistant")
st.caption("Upload a PDF and ask questions grounded in that document.")


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
    accept_multiple_files=False,
)

user_query = st.text_area(
    "Enter your question",
    height=150,
    placeholder="Ask something about the uploaded PDF",
)

ask_question = st.button("Ask AI Lawyer")


if uploaded_file is not None:
    current_file_name = uploaded_file.name
    cached_file_name = st.session_state.get("uploaded_file_name")

    if cached_file_name != current_file_name or "vector_store" not in st.session_state:
        with st.spinner("Processing PDF and preparing retrieval index..."):
            pdf_path, vector_store = get_or_create_vector_store(uploaded_file)

        st.session_state["uploaded_file_name"] = current_file_name
        st.session_state["pdf_path"] = str(pdf_path)
        st.session_state["vector_store"] = vector_store
        st.success(f"Ready to answer questions from {current_file_name}")


if ask_question:
    if uploaded_file is None:
        st.error("Upload a PDF first.")
    elif not user_query.strip():
        st.error("Enter a question before submitting.")
    else:
        st.chat_message("user").write(user_query)

        with st.spinner("Retrieving relevant passages..."):
            retrieved_docs = retrieve_docs(st.session_state["vector_store"], user_query)

        with st.spinner("Generating answer..."):
            response = answer_query(
                documents=retrieved_docs,
                model=llm_model,
                query=user_query,
            )

        st.chat_message("assistant").write(response)
