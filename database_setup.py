import pymupdf, re, chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline
import warnings
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_and_chunk_pdf(file_path, chunk_size=1000, overlap=200):
    #print(f"Reading PDF: {file_path}...")
    try:
        doc = pymupdf.open('documents/' + file_path)
    except Exception as e:
        print(f"Error opening PDF. No file 'python_book.pdf' found! ({e})")
        return []
        
    full_text = ""
    for p_num in range(len(doc)):
        page = doc.load_page(p_num)
        text = page.get_text("text")
        
        # Clean up texts
        text = re.sub(r'-\n', '', text) 
        full_text += text + "\n"
    
    # ensure texts dont get cut in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = text_splitter.split_text(full_text)
    print(f"Created {len(chunks)} smart chunks from the textbook.")
    return chunks

# Build vector database
def build_vector_database(chunks):
    if not chunks:
        print("No text chunks found.")
        return
    
    # create a folder to save the data locally
    client = chromadb.PersistentClient(path="./tutor_dbV1")
    
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    #create collection in the database
    collection = client.get_or_create_collection(
        name="python_textbook",
        embedding_function=sentence_transformer_ef
    )
    
    ids = [str(i) for i in range(len(chunks))]
    
    # add everything to the database
    collection.add(
        documents=chunks,
        ids=ids
    )
    print("Database built successfully!")
    return collection

if __name__ == "__main__":
    print("Building Database...")
    chunks = extract_and_chunk_pdf("python_book.pdf")
    build_vector_database(chunks)
