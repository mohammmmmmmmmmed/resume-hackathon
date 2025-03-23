from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
from typing import Dict, Optional
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from resume_analyzer.pdf_processor import PDFProcessor
from resume_analyzer.nlp_analysis import ResumeAnalyzer
from resume_analyzer.profile_manager import ProfileManager

app = FastAPI(title="Resume Analysis API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor = PDFProcessor()
resume_analyzer = ResumeAnalyzer()
profile_manager = ProfileManager()

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)) -> Dict:
    """Upload and analyze a resume."""
    try:
        # Read the uploaded file
        content = await file.read()
        
        # Process the PDF
        text = pdf_processor.process_pdf(content)
        if not text:
            raise HTTPException(status_code=400, detail="Failed to extract text from PDF")
            
        # Analyze the resume
        analysis = resume_analyzer.analyze_resume(text)
        
        # Generate a unique user ID
        user_id = str(uuid.uuid4())
        
        # Save the profile
        profile_manager.save_profile(user_id, analysis)
        
        return {
            "user_id": user_id,
            "profile": analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile/{user_id}")
async def get_profile(user_id: str) -> Dict:
    """Get a user's profile."""
    profile = profile_manager.get_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile

@app.put("/profile/{user_id}")
async def update_profile(user_id: str, profile_data: Dict) -> Dict:
    """Update a user's profile."""
    success = profile_manager.save_profile(user_id, profile_data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update profile")
    return {"message": "Profile updated successfully"}

@app.delete("/profile/{user_id}")
async def delete_profile(user_id: str) -> Dict:
    """Delete a user's profile."""
    success = profile_manager.delete_profile(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete profile")
    return {"message": "Profile deleted successfully"}

@app.get("/profiles")
async def get_all_profiles() -> Dict:
    """Get all profiles."""
    profiles = profile_manager.get_all_profiles()
    return {"profiles": profiles}

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"} 