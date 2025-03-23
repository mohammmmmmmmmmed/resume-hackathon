from .job_matcher.job_scraper import JobScraper

# Singleton instance of JobScraper
_job_scraper = None

def get_job_scraper() -> JobScraper:
    """Get or create a singleton instance of JobScraper."""
    global _job_scraper
    if _job_scraper is None:
        _job_scraper = JobScraper()
    return _job_scraper 