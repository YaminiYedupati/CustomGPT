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
query = None
chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    # no chat history passed
    result = chain({"question": query, "chat_history": []})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None