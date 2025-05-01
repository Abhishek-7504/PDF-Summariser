# -*- coding: utf-8 -*-
!pip install PyPDF2 transformers

# 1) Install dependencies
!pip install --quiet PyPDF2 transformers torch

# 2) Imports
import PyPDF2
from transformers import pipeline
from tqdm.notebook import tqdm

# 3) PDF → text
def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# 4) Split into chunks of ~max_len chars
def split_text(text, max_len=1000):
    words = text.split()
    chunks, curr = [], ""
    for w in words:
        if len(curr) + len(w) + 1 > max_len:
            chunks.append(curr)
            curr = w
        else:
            curr += (" " + w) if curr else w
    if curr:
        chunks.append(curr)
    return chunks

# 5) Init the distilled summarizer on GPU
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=0,               # 0 = first GPU (your T4)
    framework="pt"          # ensure PyTorch backend
)

# 6) Load PDF, chunk & summarize
pdf_path = "pdf_path"     # change to your filename
raw = extract_text_from_pdf(pdf_path)
chunks = split_text(raw, max_len=1000)

# 7) Summarize each chunk with a progress bar
summaries = []
for chunk in tqdm(chunks, desc="Summarizing"):
    out = summarizer(
        chunk,
        max_length=150,
        min_length=40,
        do_sample=False
    )
    summaries.append(out[0]["summary_text"])

# 8) Combine, print, and save
final_summary = "\n".join(summaries)
print("\n--- FINAL SUMMARY ---\n")
print(final_summary)

with open("summary_output.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)

print("\nSaved to summary_output.txt")

