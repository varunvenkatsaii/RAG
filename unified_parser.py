# parser.py
import os, re
import pymupdf 
from docx import Document
from pptx import Presentation
import asyncio
from concurrent.futures import ProcessPoolExecutor



# ---------- 1. METADATA ONLY: outline entries ----------

def meta_pdf(path):
    doc = pymupdf.open(path)
    out = [{"level": l, "title": t.strip(), "location": p}
           for l, t, p in doc.get_toc() if p >= 1]
    doc.close()
    return out

def meta_docx(path):
    out, n = [], 0
    for p in Document(path).paragraphs:
        text = p.text.strip()
        if not text:
            continue
        n += 1
        style = p.style.name or ""
        if style.startswith("Heading"):
            m = re.search(r"\d+", style)
            out.append({"level": int(m.group()) if m else 1, "title": text, "location": n})
    return out

def meta_pptx(path):
    out = []
    for n, slide in enumerate(Presentation(path).slides, start=1):
        t = slide.shapes.title
        if t is not None and t.text.strip():
            out.append({"level": 1, "title": t.text.strip(), "location": n})
    return out

def meta_txt(path):
    return []                                

META = {".pdf": meta_pdf, ".docx": meta_docx, ".pptx": meta_pptx,
        ".txt": meta_txt, ".md": meta_txt}

def extract_metadata(path):
    return META[os.path.splitext(path)[1].lower()](path)


# ---------- 2. TEXT ONLY: units with the same locations ----------
'''
def read_pdf_units(path):
    doc = pymupdf.open(path)
    units = [{"location": i, "text": p.get_text().strip()} for i, p in enumerate(doc, start=1)]
    doc.close()
    return [u for u in units if u["text"]] '''


_doc = None  # worker-side global

def _init_worker(path):
    global _doc
    _doc = pymupdf.open(path)


def _readchunkpdf( start , end):
    return  [{"location": i+1, "text": _doc[i].get_text().strip()} for i in range(start , end)]


async def readpdf_units(path , workers = 4):
    with pymupdf.open(path) as doc:
        n = len(doc)
    size = max(1, -(-n // workers))  # ceil division
    ranges = [(s, min(s + size, n)) for s in range(0, n, size)]

    loop = asyncio.get_running_loop()
    with ProcessPoolExecutor(max_workers=workers , initializer= _init_worker , initargs=(path,),) as pool:
        tasks = [
            loop.run_in_executor(pool , _readchunkpdf , s , e)
        for s , e in ranges 
        ]
        chunks = await asyncio.gather(*tasks)  # preserves order
    
    return [u for chunk in chunks for u in chunk if u["text"]]







def read_docx_units(path):
    texts = [p.text.strip() for p in Document(path).paragraphs if p.text.strip()]
    return [{"location": i, "text": t} for i, t in enumerate(texts, start=1)]

def read_pptx_units(path):
    units = []
    for n, slide in enumerate(Presentation(path).slides, start=1):
        parts = [s.text_frame.text.strip() for s in slide.shapes
                 if s.has_text_frame and s.text_frame.text.strip()]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
            parts.append("Notes: " + slide.notes_slide.notes_text_frame.text.strip())
        if parts:
            units.append({"location": n, "text": "\n".join(parts)})
    return units

def read_txt_units(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        paras = [p.strip() for p in re.split(r"\n\s*\n", f.read()) if p.strip()]
    return [{"location": i, "text": p} for i, p in enumerate(paras, start=1)]

READERS = {".pdf": readpdf_units, ".docx": read_docx_units, ".pptx": read_pptx_units,
           ".txt": read_txt_units, ".md": read_txt_units}

def read_units(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported file type: {ext}")
    units = READERS[ext](path)
    if not units:
        raise ValueError("No text found")
    return units

'''
# ---------- 3. how the chunker finds a unit's section ----------

def section_for(outline, location):
    """Heading path active at this location: last entry of each level starting at or before it."""
    stack = {}
    for o in sorted(outline, key=lambda o: o["location"]):
        if o["location"] > location:
            break
        stack[o["level"]] = o["title"]
        for l in [l for l in stack if l > o["level"]]:
            del stack[l]
    return [stack[l] for l in sorted(stack)]
'''
