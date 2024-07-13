import os
import sys
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_loaders import WebBaseLoader
import os
from langchain_groq import ChatGroq
import streamlit as st

st.title('Ask Sonar GPT')

# Step 1: Load the document from a web url
loader = WebBaseLoader("https://docs.sonarsource.com/sonarqube/latest/")
documents = loader.load()

# Step 2: Split the document into chunks with a specified chunk size
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
all_splits = text_splitter.split_documents(documents)

# Step 3: Store the document into a vector store with a specific embedding model
vectorstore = FAISS.from_documents(all_splits, HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
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

#implement PEFT

with st.form('my_form'):
    text = st.text_area('Enter text:', 'What is SonarQube?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        # few shot prompt engineering and instruction fine tuning
        prompt = f""" Provide an elaborate answer to the questions. Given are some examples. If this is the question: {prompt_1}, then the answer will be: {completion_1}.
        
        If this is the question: {prompt_2},
        
        then the answer will be: {completion_2}.
                
        If this is the question: {text}?, then the answer will be: """

        #TO-DO: add some question answer examples and calculate rouge, bert and bleu score if applicable.

        st.info(chain({"question": prompt, "chat_history": []})['answer'])