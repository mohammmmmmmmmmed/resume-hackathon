from setuptools import setup, find_packages

setup(
    name="resume_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pdfplumber>=0.10.2",
        "nltk>=3.8.1",
        "scikit-learn>=1.3.0",
        "transformers>=4.35.0",
        "sentence-transformers>=2.2.2",
        "gensim>=4.3.2",
        "dateparser>=1.1.8",
        "unidecode>=1.3.6",
        "python-Levenshtein>=0.21.1",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "watchdog>=3.0.0",
        "python-dateutil>=2.8.2"
    ],
    python_requires=">=3.8",
) 