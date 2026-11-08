from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

def create_embedding_vectorstore_splitter():
    embed_model_path = "ibm-granite/granite-embedding-small-english-r2"

    embedder = HuggingFaceEmbeddings(
        model_name=embed_model_path,
        model_kwargs={"device": "cpu"},  # Change to cuda if available
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    vector_store = Chroma(
        collection_name  = "test",
        embedding_function= embedder,
        persist_directory="./chroma_db1"
    )


    return vector_store


