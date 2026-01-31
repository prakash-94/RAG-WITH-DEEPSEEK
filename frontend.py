from rag_pipeline import answer_query, retrieve_docs, llm_model

#Phase-1 : setup UI using streamlit

#step1: setup upload pdf files
import streamlit as st

uploaded_file = st.file_uploader("Upload PDF's",
                                  type = "PDF",
                                  accept_multiple_files= False)


#step 2: chatbot skeleton (Questioning and answering)
user_query = st.text_area("Enter your Response: ", height =150, placeholder = "Ask Anything")
ask_questions = st.button("Ask AI Lawyer")

if ask_questions:

    if uploaded_file:
        st.chat_message("user").write(user_query)

        #RAG pipeline
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(documents=retrieved_docs,
                                model=llm_model, 
                                query=user_query)
       # fixed_response = "Hi, This is a General message"
        st.chat_message("AI Lawyer").write(response)
    else:
        st.error("Kindly upload PDF file first")
