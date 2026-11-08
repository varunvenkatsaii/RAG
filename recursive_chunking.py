# semantic_chunking.py  (now uses recursive splitting; keep the filename so imports still work)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

time_loaded = str(datetime.now() ) 


def default_splitter(chunk_size=1500, chunk_overlap=250):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def make_chunks(units, document_id, splitter=None):
    splitter = splitter or default_splitter()

    # one Document per unit (page / slide / paragraph), keeping its location
    docs = [
        Document(
            page_content=u["text"].strip(),
            metadata={"document_id": document_id, "page": u["location"] , "loaded" : time_loaded },
        )
        for u in units
        if u["text"].strip()
    ]

    chunks = splitter.split_documents(docs)

    # give every chunk a unique id (counter restarts on each page)
    counter = {}
    for c in chunks:
        pg = c.metadata["page"]
        i = counter.get(pg, 0)
        counter[pg] = i + 1
        c.metadata["chunk_index"] = i
        c.metadata["chunk_id"] = f"{document_id}_p{pg}_c{i}"

    print("Total chunks:", len(chunks))
    return chunks