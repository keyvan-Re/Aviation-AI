from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# ۱. خواندن تمام فایل‌های PDF از پوشه docs
print("Loading PDFs from 'docs' folder...")
# مسیر پوشه docs را اینجا می‌دهیم
loader = PyPDFDirectoryLoader("docs") 
documents = loader.load()

print(f"Loaded {len(documents)} pages in total.")

# ۲. تکه تکه کردن متن (Chunking)
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ۳. تنظیم مدل Embedding با پروکسی GapGPT
embeddings = OpenAIEmbeddings(
    openai_api_key="sk-jOn337n0y1yYP7kWTQVFRzuCsvfXlA5Y56kUDkRaQeoqhORC" , 
    openai_api_base="https://api.gapgpt.app/v1" 
    
)

# ۴. ساخت دیتابیس وکتور و ذخیره آن در یک پوشه
print("Creating Vector Database...")
vectorstore = FAISS.from_documents(docs, embeddings)

# ذخیره در پوشه ای به نام my_vector_db
vectorstore.save_local("my_vector_db") 
print("Done! Database saved to 'my_vector_db' folder.")
