from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional
from web_app.job_matcher.job_scraper import JobScraper
from web_app.dependencies import get_job_scraper
import logging

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    responses={404: {"description": "Not found"}},
)

@router.get("/list")
async def list_jobs(
    query: str = Query(None, description="Search query for jobs"),
    location: str = Query(None, description="Location to search in"),
    job_scraper: JobScraper = Depends(get_job_scraper)
) -> List[Dict]:
    """
    List all available jobs.
    
    Args:
        query: Optional search query to filter jobs
        location: Optional location to filter jobs
        
    Returns:
        List of jobs
    """
    try:
        # Validate API credentials
        if not job_scraper.adzuna_app_id or not job_scraper.adzuna_api_key:
            raise HTTPException(
                status_code=500,
                detail="Job search service is not properly configured. Please check if ADZUNA_APP_ID and ADZUNA_API_KEY environment variables are set."
            )

        # Use default query if none provided
        if not query:
            query = "software developer"
        
        # Fetch jobs
        jobs = await job_scraper.fetch_jobs(query, location)
        logger.info(f"Found {len(jobs)} jobs")
        
        return jobs
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching jobs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching jobs. Please try again later."
        )

@router.post("/match")
async def match_jobs(
    candidate_profile: Dict,
    query: Optional[str] = None,
    include_all: bool = Query(True, description="Include all jobs in response"),
    job_scraper: JobScraper = Depends(get_job_scraper)
) -> Dict[str, List[Dict]]:
    """
    Match jobs with candidate profile.
    
    Args:
        candidate_profile: Dictionary containing candidate information
        query: Optional search query to filter jobs
        include_all: Whether to include unmatched jobs in response
        
    Returns:
        Dictionary containing matched and unmatched jobs
    """
    try:
        # Validate API credentials
        if not job_scraper.adzuna_app_id or not job_scraper.adzuna_api_key:
            raise HTTPException(
                status_code=500,
                detail="Job search service is not properly configured. Please check if ADZUNA_APP_ID and ADZUNA_API_KEY environment variables are set."
            )

        # Validate candidate profile
        if not isinstance(candidate_profile, dict):
            raise HTTPException(
                status_code=400,
                detail="Invalid candidate profile format. Expected a dictionary."
            )

        # If no query provided, use candidate's top skills
        if not query and candidate_profile.get('skills'):
            query = ' '.join(candidate_profile['skills'][:3])
        elif not query:
            query = "software developer"  # Default query
        
        # Search and match jobs
        try:
            all_jobs = await job_scraper.fetch_jobs(query)
            if not all_jobs:
                logger.warning("No jobs found for the given query")
                return {"matched_jobs": [], "other_jobs": []}
                
            matched_jobs = job_scraper.match_jobs(candidate_profile, all_jobs)
            
            # Log the results
            logger.info(f"Found {len(all_jobs)} total jobs")
            
            # Split jobs into matched and unmatched based on new criteria
            jobs_with_matches = [job for job in matched_jobs if job['match_percentage'] > 0]
            jobs_without_matches = [job for job in matched_jobs if job['match_percentage'] == 0]
            
            logger.info(f"Matched {len(jobs_with_matches)} jobs")
            
            # Save matches to CSV if any found
            if jobs_with_matches:
                job_scraper.save_job_matches(
                    candidate_profile.get('contact_info', {}).get('name', 'unknown'),
                    jobs_with_matches
                )
            
            if include_all:
                return {
                    "matched_jobs": jobs_with_matches,
                    "other_jobs": jobs_without_matches
                }
            else:
                return {
                    "matched_jobs": jobs_with_matches,
                    "other_jobs": []
                }
                
        except Exception as e:
            logger.error(f"Error during job matching: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while matching jobs. Please try again later."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in match_jobs endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred. Please try again later."
        )

@router.get("/experience-levels")
def get_experience_levels() -> Dict:
    """Get mapping of experience levels to month ranges."""
    try:
        job_scraper = JobScraper()
        return job_scraper.experience_levels
    except Exception as e:
        logger.error(f"Error getting experience levels: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting experience levels: {str(e)}"
        ) 