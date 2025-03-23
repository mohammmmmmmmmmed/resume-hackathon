from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import Dict
from web_app.resume_analyzer.pdf_processor import PDFProcessor
from web_app.resume_analyzer.nlp_analysis import NLPAnalyzer
from web_app.resume_analyzer.profile_manager import ProfileManager

router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
    responses={404: {"description": "Not found"}},
)

# Initialize components
pdf_processor = PDFProcessor()
nlp_analyzer = NLPAnalyzer()
profile_manager = ProfileManager()

@router.post("/resume")
async def analyze_resume(file: UploadFile = File(...)) -> Dict:
    """
    Analyze uploaded resume and extract information.
    
    Args:
        file: Uploaded resume file (PDF)
        
    Returns:
        Dictionary containing extracted information and analysis
    """
    try:
        # Read file content
        content = await file.read()
        
        # Process PDF
        text = pdf_processor.process_pdf(content)
        
        # Analyze resume
        analysis = nlp_analyzer.analyze_resume(text)
        
        # Save profile
        profile_id = profile_manager.save_profile(analysis)
        
        return {
            "profile_id": profile_id,
            "analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{user_id}")
async def get_profile(user_id: str) -> Dict:
    """Get profile by user ID."""
    try:
        profile = profile_manager.get_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile/{user_id}")
async def update_profile(user_id: str, profile_data: Dict) -> Dict:
    """Update profile by user ID."""
    try:
        updated_profile = profile_manager.save_profile(profile_data, user_id)
        return updated_profile
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/profile/{user_id}")
async def delete_profile(user_id: str) -> Dict:
    """Delete profile by user ID."""
    try:
        profile_manager.delete_profile(user_id)
        return {"message": "Profile deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles")
async def get_all_profiles() -> Dict:
    """Get all profiles."""
    try:
        profiles = profile_manager.get_all_profiles()
        return profiles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 