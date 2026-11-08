
def is_already_loaded(document_id , vector_store):
    result = vector_store.get(
        where={"document_id": document_id},
        limit=1,
        include=[],          # we only need to know if something exists
    )
    return len(result["ids"]) > 0
