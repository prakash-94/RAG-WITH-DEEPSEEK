from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


load_dotenv()


llm_model = ChatGroq(model="llama-3.1-8b-instant")


def retrieve_docs(vector_store, query, k=4):
    return vector_store.similarity_search(query, k=k)


def get_context(documents):
    return "\n\n".join(doc.page_content for doc in documents)


custom_prompt_template = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say that you don't know.
Do not make up facts and do not answer outside the given context.

Question: {question}
Context: {context}
Answer:
"""


def answer_query(documents, model, query):
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | model
    response = chain.invoke({"question": query, "context": context})
    return response.content

