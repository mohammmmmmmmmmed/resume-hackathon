import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
import json
import os
from datetime import datetime
import time
from collections import Counter
import re
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobScraper:
    """Scrapes job postings from various sources and matches them with candidate profiles."""
    
    def __init__(self):
        # Initialize data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load API credentials from environment variables
        self.adzuna_app_id = os.getenv('ADZUNA_APP_ID')
        self.adzuna_api_key = os.getenv('ADZUNA_API_KEY')
        
        if not self.adzuna_app_id or not self.adzuna_api_key:
            logger.warning("Adzuna API credentials not found in environment variables")
        
        # Common stop words for text processing
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
            'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
            'where', 'who', 'which', 'why', 'can', 'could', 'should', 'would', 'may',
            'might', 'must', 'shall', 'into', 'if', 'then', 'else', 'than', 'too', 'very',
            'can', 'cannot', 'could', 'would', 'should', 'shall', 'will', 'might', 'must'
        }
        
        # Experience level mappings
        self.experience_levels = {
            'entry': (0, 24),  # 0-2 years
            'associate': (12, 36),  # 1-3 years
            'mid': (36, 72),  # 3-6 years
            'senior': (60, float('inf')),  # 5+ years
            'intern': (0, 12)  # 0-1 year
        }
        
        # Cache for job listings
        self.job_cache_file = os.path.join(self.data_dir, 'job_cache.json')
        self.job_cache = self._load_cache()
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _load_cache(self) -> Dict:
        """Load cached job listings."""
        if os.path.exists(self.job_cache_file):
            try:
                with open(self.job_cache_file, 'r') as f:
                    cache = json.load(f)
                # Remove old entries (older than 24 hours)
                current_time = datetime.now().timestamp()
                cache = {k: v for k, v in cache.items() 
                        if current_time - v.get('timestamp', 0) < 86400}
                return cache
            except Exception as e:
                logger.error(f"Error loading cache: {str(e)}")
                return {}
        return {}

    def _save_cache(self):
        """Save job listings to cache."""
        try:
            with open(self.job_cache_file, 'w') as f:
                json.dump(self.job_cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using regex pattern matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Replace common punctuation with spaces
        text = re.sub(r'[,.;:!?"\'\(\)\[\]{}]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out non-alphanumeric tokens and stop words
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        return tokens

    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching."""
        try:
            # Basic text cleaning
            text = text.lower()
            
            # Tokenize text
            tokens = self._tokenize_text(text)
            
            # Common technical skills and frameworks (single word)
            single_word_skills = {
                'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node',
                'sql', 'mongodb', 'postgresql', 'mysql', 'aws', 'azure', 'gcp',
                'docker', 'kubernetes', 'git', 'rest', 'api', 'microservices',
                'html', 'css', 'typescript', 'ruby', 'php', 'scala', 'hadoop',
                'spark', 'tensorflow', 'pytorch', 'ml', 'ai', 'devops', 'agile',
                'scrum', 'jira', 'linux', 'unix', 'windows', 'networking', 'numpy',
                'pandas', 'keras', 'opencv', 'nlp', 'tableau', 'statistics',
                'mathematics', 'algorithms', 'backend', 'frontend', 'fullstack',
                'ios', 'android', 'flutter', 'swift', 'kotlin'
            }
            
            # Multi-word skills to look for in the original text
            multi_word_skills = {
                'machine learning', 'deep learning', 'data science', 'data analysis',
                'data visualization', 'data engineering', 'power bi', 'ci/cd',
                'web development', 'mobile development', 'react native',
                'artificial intelligence', 'business intelligence', 'cloud computing',
                'system design', 'software architecture', 'test automation',
                'database management', 'data structures', 'computer vision',
                'natural language processing', 'time series analysis',
                'statistical analysis', 'data modeling', 'data warehousing',
                'etl processes', 'version control', 'agile methodology',
                'software development', 'full stack', 'front end', 'back end'
            }
            
            # Find single word skills
            found_skills = [token for token in tokens if token in single_word_skills]
            
            # Find multi-word skills
            for skill in multi_word_skills:
                if skill in text:
                    found_skills.append(skill)
            
            return list(set(found_skills))
            
        except Exception as e:
            logger.error(f"Error extracting skills from text: {str(e)}")
            return []

    def _determine_experience_level(self, description: str) -> str:
        """Determine the experience level required from job description."""
        try:
            description = description.lower()
            
            # Keywords for different experience levels
            level_keywords = {
                'entry': ['entry level', 'junior', 'graduate', '0-2 years', 'fresh graduate'],
                'associate': ['associate', '1-3 years', '2-3 years'],
                'mid': ['mid level', 'intermediate', '3-5 years', '4-6 years'],
                'senior': ['senior', 'lead', '5+ years', '6+ years', 'principal'],
                'intern': ['intern', 'internship', 'trainee']
            }
            
            # Count occurrences of keywords for each level
            level_scores = {level: 0 for level in level_keywords}
            for level, keywords in level_keywords.items():
                for keyword in keywords:
                    if keyword in description:
                        level_scores[level] += 1
            
            # Return the level with highest score, default to 'entry' if no clear indication
            max_score = max(level_scores.values())
            if max_score == 0:
                return 'entry'
            
            return max(level_scores.items(), key=lambda x: x[1])[0]
            
        except Exception as e:
            logger.error(f"Error determining experience level: {str(e)}")
            return 'entry'  # Default to entry level on error

    async def scrape_adzuna(self, query: str, location: str = None) -> List[Dict]:
        """Scrape jobs from Adzuna API."""
        try:
            if not self.adzuna_app_id or not self.adzuna_api_key:
                logger.error("Adzuna API credentials not found")
                return []
            
            # Base URL for Adzuna API
            base_url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
            
            # Build query parameters
            params = {
                'app_id': self.adzuna_app_id,
                'app_key': self.adzuna_api_key,
                'results_per_page': 50,  # Adjust as needed
                'what': query,
                'content-type': 'application/json'
            }
            
            # Add location if provided
            if location:
                params['where'] = location
            
            async with aiohttp.ClientSession() as session:
                async with session.get(base_url, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Adzuna API error: {response.status} - {error_text}")
                        return []
                    
                    data = await response.json()
                    
                    if not data.get('results'):
                        logger.warning(f"No jobs found for query: {query} in location: {location}")
                        return []
                    
                    jobs = []
                    for job in data['results']:
                        processed_job = {
                            'title': job.get('title', ''),
                            'company': job.get('company', {}).get('display_name', 'Unknown'),
                            'location': job.get('location', {}).get('display_name', 'Remote'),
                            'description': job.get('description', ''),
                            'url': job.get('redirect_url', ''),
                            'salary_min': job.get('salary_min'),
                            'salary_max': job.get('salary_max'),
                            'source': 'Adzuna',
                            'id': f"adzuna_{hash(job.get('redirect_url', ''))}"  # Add ID during processing
                        }
                        jobs.append(processed_job)
                    
                    logger.info(f"Found {len(jobs)} jobs from Adzuna for query: {query}")
                    return jobs
                    
        except aiohttp.ClientError as e:
            logger.error(f"Network error while fetching from Adzuna: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error scraping Adzuna: {str(e)}")
            return []

    async def scrape_github_jobs(self, query: str) -> List[Dict]:
        """Scrape jobs from GitHub Jobs."""
        try:
            url = f"https://jobs.github.com/positions.json"
            params = {'description': query}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            jobs = response.json()
            
            return [{
                'title': job.get('title'),
                'company': job.get('company'),
                'location': job.get('location'),
                'description': job.get('description'),
                'url': job.get('url'),
                'source': 'GitHub Jobs',
                'timestamp': datetime.now().timestamp()
            } for job in jobs]
            
        except Exception as e:
            logger.error(f"Error scraping GitHub Jobs: {str(e)}")
            return []

    async def fetch_jobs(self, query: str, location: str = None) -> List[Dict]:
        """Fetch jobs from various sources without matching."""
        # Check cache first
        cache_key = f"{query}_{location}_{datetime.now().strftime('%Y%m%d')}"
        if cache_key in self.job_cache:
            return self.job_cache[cache_key]['jobs']

        # Scrape jobs from multiple sources
        jobs = []
        
        # Fetch from Adzuna
        adzuna_jobs = await self.scrape_adzuna(query, location)
        for job in adzuna_jobs:
            job['id'] = f"adzuna_{hash(job['url'])}"  # Add unique ID
            jobs.append(job)
        
        # Fetch from GitHub Jobs
        github_jobs = await self.scrape_github_jobs(query)
        for job in github_jobs:
            job['id'] = f"github_{hash(job['url'])}"  # Add unique ID
            jobs.append(job)
        
        # Update cache
        self.job_cache[cache_key] = {
            'jobs': jobs,
            'timestamp': datetime.now().timestamp()
        }
        self._save_cache()
        
        return jobs

    async def search_jobs(self, query: str, candidate_profile: Dict) -> List[Dict]:
        """Search and match jobs for a candidate."""
        # Fetch all jobs
        jobs = await self.fetch_jobs(query)
        
        # Match jobs with candidate profile
        matched_jobs = self.match_jobs(candidate_profile, jobs)
        
        return matched_jobs

    def match_jobs(self, candidate_profile: Dict, jobs: List[Dict]) -> List[Dict]:
        """Match jobs with candidate profile based on skills and experience."""
        candidate_months = candidate_profile.get('total_experience', {}).get('total_months', 0)
        candidate_skills = set(candidate_profile.get('skills', []))
        
        matched_jobs = []
        for job in jobs:
            # Extract required skills from job description
            job_skills = set(self._extract_skills_from_text(job['description']))
            
            # Calculate skill match percentage
            total_required_skills = len(job_skills)
            matching_skills = len(candidate_skills.intersection(job_skills))
            missing_skills = job_skills - candidate_skills
            missing_skills_count = len(missing_skills)
            
            if total_required_skills > 0:
                match_percentage = (matching_skills / total_required_skills) * 100
            else:
                match_percentage = 0
            
            # Determine experience level requirement
            required_level = self._determine_experience_level(job['description'])
            min_months, max_months = self.experience_levels[required_level]
            
            # Check if candidate meets experience requirement
            experience_match = min_months <= candidate_months <= max_months
            
            job_match = {
                **job,
                'match_percentage': round(match_percentage, 2),
                'matching_skills': list(candidate_skills.intersection(job_skills)),
                'missing_skills': list(missing_skills),
                'experience_match': experience_match,
                'required_experience_level': required_level
            }
            
            # Add to matched jobs if:
            # 1. At least 50% skill match OR
            # 2. Only 2-3 skills missing
            if match_percentage >= 50 or (total_required_skills > 0 and missing_skills_count <= 3):
                matched_jobs.append(job_match)
            else:
                # Add to other jobs
                job_match['match_percentage'] = 0  # Reset match percentage for unmatched jobs
                matched_jobs.append(job_match)
        
        # Sort by match percentage (descending) and then by number of missing skills (ascending)
        return sorted(matched_jobs, key=lambda x: (x['match_percentage'], -len(x['missing_skills'])), reverse=True)

    def save_job_matches(self, candidate_name: str, matched_jobs: List[Dict]):
        """Save job matches to CSV."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', candidate_name).lower()
        csv_path = os.path.join(self.data_dir, f'job_matches_{clean_name}_{timestamp}.csv')
        
        # Prepare data for DataFrame
        data = []
        for job in matched_jobs:
            data.append({
                'Title': job['title'],
                'Company': job['company'],
                'Location': job['location'],
                'Match Percentage': job['match_percentage'],
                'Experience Match': job['experience_match'],
                'Required Level': job['required_experience_level'],
                'Missing Skills': ', '.join(job['missing_skills']),
                'URL': job['url'],
                'Source': job['source']
            })
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        logger.info(f"Job matches saved to {csv_path}") 