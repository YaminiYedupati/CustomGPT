import os
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, OnlinePDFLoader
from langchain_groq import ChatGroq
import requests
import streamlit as st

st.title('Ask Sonar GPT')

# Function to fetch HTML content including text, pdf, doc contents of a webpage
def retrieve_contents_from_page(url):
    doc_splits = []
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to load web page")
    soup = BeautifulSoup(response.content, 'html.parser')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    loader = WebBaseLoader(url)
    documents = loader.load()
    split_docs = text_splitter.split_documents(documents)
    for doc in split_docs:
        doc_splits.append(doc)
    links = soup.find_all('a', href=lambda href: href and href.__contains__(".pdf"))
    for pdf in links:
        loader = OnlinePDFLoader(pdf)
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        for doc in split_docs:
            doc_splits.append(doc)
    return doc_splits

# Function to parse HTML and extract links
def extract_links(base_url):
    links = []
    response = requests.get(base_url)
    if response.status_code != 200:
        print("Failed to load web page")
    soup = BeautifulSoup(response.content, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and href.startswith('/') or base_url in href:
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
    return links

# Main function to retrieve texts and file contents from all subpages and store all of them in dictionary.
def retrieve_contents_from_subpages(main_url):
    all_doc_splits = []
    doc_splits = retrieve_contents_from_page(main_url)
    for split in doc_splits:
        all_doc_splits.append(split)
    base_url = main_url.rstrip('/')
    
    subpage_links = extract_links(base_url)
    print(f"Found {len(subpage_links)} subpage(s).")
    
    for _, subpage_link in enumerate(subpage_links, 1):
        doc_splits = retrieve_contents_from_page(subpage_link)
        for split in doc_splits:
            all_doc_splits.append(split)
    return all_doc_splits
    
if __name__ == "__main__":

    # Load the document from a web url
    url = "https://docs.sonarsource.com/sonarqube/latest/"
    doc_splits = retrieve_contents_from_subpages(url)
    print(doc_splits)
    
    # Store the document into a vector store with a specific embedding model
    vectorstore = FAISS.from_documents(doc_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

    #vectorstore = FAISS.from_documents(doc_splits, HFEmbeddingsModel)
    os.environ["GROQ_API_KEY"] = "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"
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

    # Query against your own data
    chain = ConversationalRetrievalChain.from_llm(llm,
                                                vectorstore.as_retriever(),
                                                return_source_documents=True)

    prompt_1 = "What is SonarQube?"
    completion_1 = "SonarQube is an open-source platform developed by SonarSource for continuous inspection of code quality to perform automatic reviews with static analysis of code to detect bugs, code smells, and security vulnerabilities in over 20 programming languages. It provides reports on code coverage, code duplication, complexity, potential bugs, and more, helping development teams to maintain code quality throughout the development and maintenance phases of their projects."
    prompt_2 = "How can I integrate SonarQube with my application? Provide a very concise answer"
    completion_2 = """Integrating SonarQube with your application typically involves these steps:

            Install SonarQube: Set up SonarQube on a server or cloud instance.

            Configure Your Build: Integrate SonarQube into your build process (e.g., Maven, Gradle, Jenkins) using plugins or build scripts.

            Run Analysis: Execute SonarQube analysis during your build to generate code quality reports.

            View Results: Access and interpret the reports in the SonarQube dashboard to improve code quality.

            Each step may require specific configuration based on your development environment and tools."""

    with st.form('my_form'):
        text = st.text_area('Enter text:', 'What is SonarQube?')
        submitted = st.form_submit_button('Submit')
        if submitted:
            # few shot prompt engineering and instruction fine tuning - included a RAG prompt from rlm/rag-prompt
            # aware of the context window
            prompt = f""" You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Provide an elaborate answer to the questions. Given are some examples. If this is the question: {prompt_1}, then the answer will be: {completion_1}.
            
            If this is the question: {prompt_2},
            
            then the answer will be: {completion_2}.
                    
            If this is the question: {text}?, then the answer will be: """

            #TO-DO: add some question answer examples and calculate rouge, bert and bleu score if applicable.

            st.info(chain({"question": prompt, "chat_history": []})['answer'])