from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# 1. Set up Gemini API Key
os.environ["GOOGLE_API_KEY"] = "***************************************"  # paste your key here

# 2. Load PDF
loader = PyPDFLoader(r"Paste Your Document path ")
docs = loader.load()

# 3. Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
splits = text_splitter.split_documents(docs)

# 4. Create FAISS vector store using Gemini embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vectorstore = FAISS.from_documents(splits, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 6. Load Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# 7. Connect Gemini with RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 8. Example query
query = "What example does Rich Dad use to explain how taxes and corporations give the rich an advantage?"
answer = qa_chain.run(query)
print("\n Answer:", answer)
