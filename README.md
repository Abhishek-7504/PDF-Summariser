A Python script that extracts text from any PDF and generates an abstractive summary using the "BART" model.
The model is downloaded from Hugging Face under Apache 2.0.

Features:

Uses [PyPDF2] to extract text page by page.

Chunking: Splits long documents into ~1,000-character segments to fit model input limits.

Abstractive Summaries: Powered by sshleifer/distilbart-cnn-12-6 via Hugging Face Transformers.

Progress Bars: Visualize progress with tqdm.

Output: Combines chunk summaries and saves to summary_output.txt
