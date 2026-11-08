import hashlib

def create_docid(path):
    document_id = hashlib.sha256(
        path.encode("utf-8")
    ).hexdigest()[:12]
    return document_id