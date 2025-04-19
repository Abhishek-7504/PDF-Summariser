A lightweight Python script that extracts text from any PDF and generates an abstractive summary using the distilled BART model.
The model is downloaded from Hugging Face under Apache 2.0, and that anyone who wants to use Hugging Face’s hosted API will need to supply their own token and abide by HF’s rate limits and billing.

Features
-PDF → Text: Uses [PyPDF2] to extract text page by page.
-Chunking: Splits long documents into ~1,000-character segments to fit model input limits.
-Abstractive Summaries: Powered by sshleifer/distilbart-cnn-12-6 via Hugging Face Transformers.
-GPU‑Ready: Leverages Colab’s T4 GPU (device=0) for significantly faster inference.
-Progress Bars: Visualize progress with tqdm.
-Output: Combines chunk summaries and saves to summary_output.txt
