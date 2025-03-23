import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from typing import Dict, List, Tuple, Optional
import re
import logging
from datetime import datetime
import pandas as pd
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

class NLPAnalyzer:
    """Analyzes resume text using NLP techniques."""
    
    def __init__(self):
        # Initialize data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def analyze_resume(self, text: str) -> dict:
        """Analyze resume text and extract structured information."""
        try:
            # Log the input text
            logger.info("Input text:")
            logger.info(text[:200] + "..." if len(text) > 200 else text)
            logger.info("-" * 80)
            
            # Extract sections
            sections = self._split_into_sections(text)
            logger.info("Found sections: %s", list(sections.keys()))
            
            # Log each section's content (shortened for clarity)
            for section, content in sections.items():
                logger.info(f"\nSection '{section}':")
                logger.info(content[:100] + "..." if len(content) > 100 else content)
                logger.info("-" * 40)
            
            # Extract information from each section
            contact_info = self._extract_contact_info(sections.get('HEADER', '') or sections.get('CONTACT', '') or '')
            
            # If name wasn't detected in contact info, try to get it from the beginning of the resume
            if not contact_info["name"]:
                lines = text.strip().split('\n')
                if lines and lines[0].strip():
                    contact_info["name"] = lines[0].strip()
            
            education = self._extract_education(sections.get('EDUCATION', ''))
            experience = self._extract_experience(sections.get('EXPERIENCE', ''))
            skills = self._extract_skills(
                sections.get('TECHNICAL SKILLS', '') or 
                sections.get('SKILLS', '') or 
                sections.get('ADDITIONAL SKILLS', '') or 
                ''
            )
            
            # Calculate total experience
            total_experience = self._calculate_total_experience(experience)
            
            # Log extracted information
            logger.info("\nExtracted information:")
            logger.info(f"Contact info: {contact_info}")
            logger.info(f"Education: {education}")
            logger.info(f"Experience: {experience}")
            logger.info(f"Total experience: {total_experience}")
            logger.info(f"Skills: {skills}")
            
            # Create profile data
            profile_data = {
                "contact_info": contact_info,
                "education": education,
                "experience": experience,
                "total_experience": total_experience,
                "skills": skills
            }
            
            # Save to CSV
            self._save_to_csv(profile_data)
            
            return profile_data
            
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            return {
                "contact_info": {
                    "name": "",
                    "email": "",
                    "phone": "",
                    "location": "",
                    "website": "",
                    "linkedin": ""
                },
                "education": [],
                "experience": [],
                "total_experience": {
                    "total_months": 0,
                    "years": 0,
                    "remaining_months": 0,
                    "formatted": "0 months"
                },
                "skills": []
            }

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split resume text into sections with improved section detection."""
        sections = {}
        current_section = "HEADER"
        current_content = []
        
        # Split text into lines and remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Section header detection regex
        section_header_pattern = r'^[A-Z][A-Z\s&]+$'
        
        # Common section names in resumes (case-insensitive for flexibility)
        common_sections = [
            'EDUCATION', 'EXPERIENCE', 'SKILLS', 'TECHNICAL SKILLS', 'PROJECTS', 
            'AWARDS', 'CERTIFICATIONS', 'OBJECTIVE', 'SUMMARY', 'LANGUAGES',
            'CONTACT', 'ADDITIONAL', 'INTERESTS'
        ]
        
        # Process lines
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this is a section header
            is_header = False
            if re.match(section_header_pattern, line):
                # Check if contains common section keywords
                for section in common_sections:
                    if section in line.upper():
                        # Save previous section content
                        if current_content:
                            sections[current_section] = '\n'.join(current_content)
                            current_content = []
                        
                        # Start new section
                        current_section = line.upper()
                        is_header = True
                        break
            
            # If not a header, add to current section content
            if not is_header:
                current_content.append(line)
            
            i += 1
        
        # Save the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def _extract_contact_info(self, text: str) -> dict:
        """Extract contact information from text with improved pattern matching."""
        contact_info = {
            "name": "",
            "email": "",
            "phone": "",
            "location": "",
            "website": "",
            "linkedin": ""
        }
        
        # Patterns for contact information
        patterns = {
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'phone': r'(?:\+\d{1,3}[-.\s]?)?\d{10}',
            'linkedin': r'(?:linkedin\.com/in/|linkedin|lujayn)',
            'website': r'(?:https?://)?(?:www\.)?\S+\.(?:com|io|dev|app|vercel)',
            'location': r'(?:Kerala|India|Ernakulam|Cochin|Kochi)',
            'name': r'^([A-Z][A-Z\s]+)$',
        }
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Try to extract name from first line if it matches name pattern
        if lines and re.match(patterns['name'], lines[0]):
            contact_info["name"] = lines[0]
        
        # Process all lines for contact information
        for line in lines:
            # Split line by common separators
            parts = re.split(r'[⋄,|*@ï\n]', line)
            for part in parts:
                part = part.strip()
                
                # Check for email
                email_match = re.search(patterns['email'], part)
                if email_match and not contact_info["email"]:
                    contact_info["email"] = email_match.group(0)
                
                # Check for phone
                phone_match = re.search(patterns['phone'], part)
                if phone_match and not contact_info["phone"]:
                    contact_info["phone"] = phone_match.group(0)
                
                # Check for LinkedIn
                linkedin_match = re.search(patterns['linkedin'], part.lower())
                if linkedin_match and not contact_info["linkedin"]:
                    # Try to extract full LinkedIn URL if present
                    full_linkedin = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+', part)
                    if full_linkedin:
                        contact_info["linkedin"] = full_linkedin.group(0)
                    else:
                        # Just store the username if that's all we have
                        username_match = re.search(r'linkedin\.com/in/([\w-]+)', part)
                        if username_match:
                            contact_info["linkedin"] = f"linkedin.com/in/{username_match.group(1)}"
                        else:
                            # Check for LinkedIn indicators
                            if "linkedin" in part.lower():
                                for word in part.split():
                                    if "linkedin" not in word.lower() and len(word) > 3:
                                        contact_info["linkedin"] = f"linkedin.com/in/{word.lower()}"
                    break
                
                # Check for website
                website_match = re.search(patterns['website'], part)
                if website_match and not contact_info["website"] and 'linkedin.com' not in part:
                    contact_info["website"] = website_match.group(0)
                
                # Check for location indicators
                for location in ['Ernakulam', 'Kerala', 'India', 'Cochin', 'Kochi']:
                    if location in part:
                        # Extract the complete location phrase
                        if not contact_info["location"]:
                            location_parts = []
                            for loc_part in part.split(','):
                                if any(loc in loc_part for loc in ['Ernakulam', 'Kerala', 'India', 'Cochin', 'Kochi']):
                                    location_parts.append(loc_part.strip())
                            if location_parts:
                                contact_info["location"] = ', '.join(location_parts)
            
        return contact_info
    
    def _extract_education(self, text: str) -> list:
        """Extract education information from text with improved pattern matching."""
        education = []
        
        # Patterns for education information
        patterns = {
            'degree': r'(?:B\.Tech|M\.Tech|Bachelor|Master|Ph\.D|MBA|Higher Secondary Education|Secondary Education)',
            'institution': r'(?:University|College|School|Institute|CUSAT)',
            'year': r'(?:20\d{2}|19\d{2})',
            'date_range': r'(?:20\d{2})\s*(?:–|-|\s)\s*(?:Present|20\d{2})',
            'percentage': r'Achieved\s+(\d+\.?\d*)%',
        }
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Store potential education segments
        education_segments = []
        current_segment = []
        
        # First, identify education segments
        for line in lines:
            # Check if this line likely starts a new education entry
            if (re.search(patterns['degree'], line) or 
                re.search(patterns['institution'], line) or 
                re.search(patterns['date_range'], line)):
                
                # Save previous segment if it exists
                if current_segment:
                    education_segments.append('\n'.join(current_segment))
                    current_segment = []
                
                # Start new segment
                current_segment.append(line)
            elif current_segment:
                # Continue existing segment
                current_segment.append(line)
        
        # Add the last segment
        if current_segment:
            education_segments.append('\n'.join(current_segment))
        
        # Process each education segment
        for segment in education_segments:
            segment_lines = segment.split('\n')
            
            entry = {
                "degree": "",
                "institution": "",
                "end_date": "",
                "coursework": []
            }
            
            # Extract degree
            for line in segment_lines:
                degree_match = re.search(patterns['degree'], line)
                if degree_match:
                    # Get the full degree (including "in X" part if present)
                    full_degree = line
                    if "in " in line[degree_match.end():]:
                        field_part = line[degree_match.end():].split(',')[0]
                        entry["degree"] = (degree_match.group(0) + field_part).strip()
                    else:
                        entry["degree"] = degree_match.group(0)
                    break
            
            # Extract institution
            for line in segment_lines:
                institution_match = re.search(patterns['institution'], line)
                if institution_match:
                    # Get the full institution name
                    if "University" in line or "College" in line or "School" in line or "Institute" in line or "CUSAT" in line:
                        parts = line.split(',')
                        for part in parts:
                            if ("University" in part or "College" in part or "School" in part or "Institute" in part or "CUSAT" in part):
                                entry["institution"] = part.strip()
                                break
                    # If no specific institution identifier, take the full line
                    if not entry["institution"]:
                        entry["institution"] = line.split(',')[0].strip()
                    break
            
            # Extract date
            for line in segment_lines:
                # Try to find date range pattern
                date_range_match = re.search(patterns['date_range'], line)
                if date_range_match:
                    dates = date_range_match.group(0).split('–')
                    if len(dates) == 1:
                        dates = date_range_match.group(0).split('-')
                    if len(dates) == 1:
                        dates = date_range_match.group(0).split(' ')
                    
                    if len(dates) >= 2:
                        end_date = dates[1].strip()
                    else:
                        end_date = dates[0].strip()
                        
                    if end_date != "Present":
                        entry["end_date"] = re.search(r'20\d{2}', end_date).group(0) if re.search(r'20\d{2}', end_date) else ""
                    else:
                        entry["end_date"] = "Present"
                    break
                
                # Try to find single year
                year_match = re.search(patterns['year'], line)
                if year_match:
                    entry["end_date"] = year_match.group(0)
                    break
            
            # Extract percentage if available
            for line in segment_lines:
                percentage_match = re.search(patterns['percentage'], line)
                if percentage_match:
                    entry["coursework"].append(f"Achieved {percentage_match.group(1)}%")
                elif "Achieved" in line and "%" in line:
                    percentage = re.search(r'(\d+\.?\d*)%', line)
                    if percentage:
                        entry["coursework"].append(f"Achieved {percentage.group(1)}%")
            
            # Add coursework from bullet points
            for line in segment_lines:
                if line.strip().startswith('•') or line.strip().startswith('-'):
                    entry["coursework"].append(line.lstrip('•-').strip())
            
            # Add to education list if we have at least degree or institution
            if entry["degree"] or entry["institution"]:
                education.append(entry)
                    
        return education
    
    def _extract_experience(self, text: str) -> list:
        """Extract work experience from text with improved pattern matching."""
        experience = []
        
        # Patterns for experience information
        patterns = {
            'position_date_standard': r'^(.*?)\s+([A-Z][a-z]{2}\s+\d{4})\s*-\s*(Present|[A-Z][a-z]{2}\s+\d{4})',
            'position_date_z': r'^([^z]+)z\s*([A-Z][a-z]{2}\s+\d{4}|\d{4}(?:\s*[–-]\s*(?:Present|\d{4})?)?)\s*(?:\*\s*([^•]+))?',
            'bullet': r'^\s*[•●■-]\s*(.+)',
        }
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        current_entry = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines or single bullet points
            if not line or line in ['•', '-']:
                i += 1
                continue
            
            # Check for bullet points
            bullet_match = re.match(patterns['bullet'], line)
            if bullet_match and current_entry:
                current_entry["description"].append(bullet_match.group(1))
                i += 1
                continue
            
            # Try standard format (Position Mar 2025 - Present)
            standard_match = re.match(patterns['position_date_standard'], line)
            if standard_match:
                if current_entry:
                    experience.append(current_entry)
                
                position = standard_match.group(1).strip()
                start_date = standard_match.group(2)
                end_date = standard_match.group(3)
                
                # Get company and location from next line
                company = ""
                location = ""
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    if ',' in next_line:
                        parts = next_line.split(',')
                        company = parts[0].strip()
                        location = ','.join(parts[1:]).strip()
                    else:
                        company = next_line.strip()
                
                current_entry = {
                    "position": position,
                    "company": company,
                    "location": location,
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": []
                }
                i += 2  # Skip next line as we've used it for company
                continue
            
            # Try z-format (Student Intern z May 2024 * Cochin, Kerala)
            z_match = re.match(patterns['position_date_z'], line)
            if z_match:
                if current_entry:
                    experience.append(current_entry)
                
                position = z_match.group(1).strip()
                date_str = z_match.group(2).strip()
                location = z_match.group(3).strip() if z_match.group(3) else ""
                
                # Parse the date string
                start_date = ""
                end_date = ""
                if '–' in date_str or '-' in date_str:
                    parts = re.split(r'[–-]', date_str)
                    start_date = parts[0].strip()
                    end_date = parts[1].strip() if len(parts) > 1 else "Present"
                else:
                    start_date = date_str
                    end_date = date_str
                
                # Get company from next line
                company = ""
                if i + 1 < len(lines) and not re.match(patterns['bullet'], lines[i + 1]):
                    company = lines[i + 1].strip()
                    i += 1  # Skip next line as we've used it
                
                current_entry = {
                    "position": position,
                    "company": company,
                    "location": location,
                    "start_date": start_date,
                    "end_date": end_date,
                    "description": []
                }
            
            i += 1
        
        # Add the last entry if exists
        if current_entry:
            experience.append(current_entry)
                    
        return experience
    
    def _extract_skills(self, text: str) -> list:
        """Extract skills from text with improved pattern matching."""
        skills = []
        
        # Check if text is empty
        if not text:
            return skills
        
        # Split into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        for line in lines:
            # Skip headers
            if all(c.isupper() or c.isspace() for c in line):
                continue
                
            # Check for category: skills format
            if ':' in line:
                skills_text = line.split(':', 1)[1].strip()
                
                # Extract skills from comma-separated list
                if ',' in skills_text:
                    skills.extend([skill.strip() for skill in skills_text.split(',') if skill.strip()])
                else:
                    # Single skill
                    skills.append(skills_text)
            elif ',' in line:
                # Comma-separated list without category
                skills.extend([skill.strip() for skill in line.split(',') if skill.strip()])
            elif line.startswith('•') or line.startswith('-'):
                # Bullet point
                skills.append(line.lstrip('•-').strip())
            else:
                # Single skill on a line
                skills.append(line)
        
        # Remove duplicates and empty strings
        return list(set(filter(None, skills)))

    def _calculate_total_experience(self, experience: list) -> dict:
        """Calculate total months of experience from the experience list."""
        total_months = 0
        
        # Dictionary to convert month names to numbers
        month_to_num = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        
        # Current date for "Present" calculations
        current_date = datetime.now()
        
        for entry in experience:
            try:
                start_date_str = entry.get('start_date', '')
                end_date_str = entry.get('end_date', '')
                
                if not start_date_str:
                    continue
                
                # Handle "Present" in end date
                if end_date_str.lower() == 'present':
                    end_month = current_date.month
                    end_year = current_date.year
                elif not end_date_str:
                    # If no end date, assume it's the same as start date
                    end_date_str = start_date_str
                
                # Parse start date
                start_month = None
                start_year = None
                
                # Try to extract month and year from start date
                # Format: "Mon YYYY"
                month_year_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', start_date_str)
                if month_year_match:
                    start_month = month_to_num[month_year_match.group(1)]
                    start_year = int(month_year_match.group(2))
                else:
                    # Format: "YYYY"
                    year_match = re.search(r'(\d{4})', start_date_str)
                    if year_match:
                        start_month = 1  # Assume January if only year is specified
                        start_year = int(year_match.group(1))
                
                # Parse end date
                end_month = None
                end_year = None
                
                if end_date_str.lower() == 'present':
                    end_month = current_date.month
                    end_year = current_date.year
                else:
                    # Format: "Mon YYYY"
                    month_year_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})', end_date_str)
                    if month_year_match:
                        end_month = month_to_num[month_year_match.group(1)]
                        end_year = int(month_year_match.group(2))
                    else:
                        # Format: "YYYY"
                        year_match = re.search(r'(\d{4})', end_date_str)
                        if year_match:
                            end_month = 12  # Assume December if only year is specified
                            end_year = int(year_match.group(1))
                
                # Calculate months if we have all the necessary data
                if start_month and start_year and end_month and end_year:
                    months = (end_year - start_year) * 12 + (end_month - start_month) + 1  # Include the current month
                    total_months += max(0, months)  # Ensure we don't add negative months
                
            except (KeyError, ValueError, IndexError, AttributeError) as e:
                logger.warning(f"Error calculating experience for entry: {entry}. Error: {str(e)}")
                continue
        
        # Convert to years and months
        years = total_months // 12
        remaining_months = total_months % 12
        
        return {
            "total_months": total_months,
            "years": years,
            "remaining_months": remaining_months,
            "formatted": f"{years} years {remaining_months} months" if years > 0 else f"{remaining_months} months"
        }

    def _save_to_csv(self, profile_data: dict):
        """Save profile data to CSV file."""
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Get person's name and clean it for filename
        name = profile_data['contact_info']['name']
        if not name:
            name = "unknown_candidate"
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
        
        # Create filename with timestamp and name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.data_dir, f'{clean_name}_{timestamp}.csv')
        
        # Create DataFrame from profile data
        df = pd.DataFrame({
            'Name': [profile_data['contact_info']['name']],
            'Email': [profile_data['contact_info']['email']],
            'Phone': [profile_data['contact_info']['phone']],
            'Location': [profile_data['contact_info']['location']],
            'Website': [profile_data['contact_info']['website']],
            'LinkedIn': [profile_data['contact_info']['linkedin']],
            'Total Experience': [profile_data['total_experience']['formatted']],
            'Skills': [', '.join(profile_data['skills'])]
        })
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Profile data saved to {csv_path}") 