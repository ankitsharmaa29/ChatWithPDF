import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def pick_gemini_model() -> str:
    """
    Prioritize Gemini-1.5 Flash or Alpha. 
    Fallback to any other available chat model.
    """

    preferred = [
        "models/gemini-1.5-flash",
        "models/gemini-1.5-alpha",
        "models/gemini-pro",          
        "models/chat-bison-001",      
        "models/text-bison-001",
    ]
    available = [m.name for m in genai.list_models()]
    for name in preferred:
        if name in available:
            return name
    raise RuntimeError(
        "No supported chat model found. "
        "Enable the Generative Language API or request access to Gemini-1.5."
    )

CHAT_MODEL_NAME = pick_gemini_model()


def extract_pdf_text(files) -> str:
    text = ""
    for pdf in files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            pg = page.extract_text()
            if pg:
                text += pg
    return text

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10_000, chunk_overlap=1_000)
    return splitter.split_text(text)

def build_vector_store(chunks):
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = FAISS.from_texts(chunks, emb)
    vs.save_local("faiss_index")

def load_vector_store():
    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)

def make_qa_chain():
    prompt = PromptTemplate(
        template=(
            "Answer as thoroughly as possible from the given context.\n"
            'If the answer is not in the context, reply exactly:\n'
            '"answer is not available in the context"\n\n'
            "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        ),
        input_variables=["context", "question"],
    )
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, temperature=0.3)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def answer_question(q: str):
    store = load_vector_store()
    docs = store.similarity_search(q)
    chain = make_qa_chain()
    result = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
    st.write("**Reply:**", result["output_text"])


def main():
    st.set_page_config(page_title="Chat-with-PDF")
    st.header(f"PDF Q&A • Using model `{CHAT_MODEL_NAME}`")

    query = st.text_input("Ask a question about the uploaded PDFs:")
    if query:
        answer_question(query)

    with st.sidebar:
        st.title("Upload & Index PDFs")
        pdfs = st.file_uploader(
            "Select PDF file(s) then click **Process**", type=["pdf"], accept_multiple_files=True
        )
        if st.button("Process"):
            if not pdfs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Extracting & indexing…"):
                    raw = extract_pdf_text(pdfs)
                    chunks = chunk_text(raw)
                    build_vector_store(chunks)
                st.success("Index built! Now ask your questions above.")

if __name__ == "__main__":
    main()
