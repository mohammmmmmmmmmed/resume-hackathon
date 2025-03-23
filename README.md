# Resume Analyzer

A powerful resume analysis system that processes PDF resumes, extracts information using advanced NLP, and generates detailed profiles with ratings.

## Features

- PDF Resume Upload and Processing
- Advanced NLP Analysis
- Profile Generation and Rating
- Interactive Profile Editing
- CSV Data Management
- Real-time Updates

## Project Structure

```
resume-analyzer/
├── resume_analyzer/          # Core library
│   ├── pdf_processor/       # PDF processing module
│   ├── nlp_analyzer/        # NLP analysis module
│   └── profile_generator/   # Profile generation module
├── web_app/                 # Web interface
│   ├── frontend/           # React/Next.js frontend
│   └── backend/            # FastAPI backend
├── data/                   # Data storage
│   ├── profiles/          # User profiles
│   └── templates/         # CSV templates
└── tests/                 # Test suite
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLP models:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the backend server:
```bash
cd web_app/backend
uvicorn main:app --reload
```

2. Start the frontend development server:
```bash
cd web_app/frontend
npm install
npm run dev
```

3. Access the application at `http://localhost:3000`

## Development

- Run tests: `pytest`
- Format code: `black .`
- Check types: `mypy .`

## License

MIT License 