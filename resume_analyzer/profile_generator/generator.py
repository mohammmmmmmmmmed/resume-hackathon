from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileGenerator:
    """Handles profile generation and CSV management."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.profiles_dir = self.data_dir / "profiles"
        self.templates_dir = self.data_dir / "templates"
        
        # Create necessary directories
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize profile template
        self._create_template()
    
    def _create_template(self):
        """Create the profile template CSV if it doesn't exist."""
        template_path = self.templates_dir / "profile_template.csv"
        if not template_path.exists():
            template_data = {
                'section': ['personal', 'skills', 'experience', 'education'],
                'field': ['name', 'email', 'phone', 'technical_skills', 'soft_skills', 
                         'company', 'position', 'duration', 'institution', 'degree'],
                'value': ['', '', '', '', '', '', '', '', '', ''],
                'confidence_score': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'last_updated': ['', '', '', '', '', '', '', '', '', '']
            }
            pd.DataFrame(template_data).to_csv(template_path, index=False)
    
    def create_profile(self, user_id: str, analysis_results: Dict) -> bool:
        """
        Create a new profile from analysis results.
        
        Args:
            user_id: Unique identifier for the user
            analysis_results: Dictionary containing analysis results
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create user directory
            user_dir = self.profiles_dir / user_id
            user_dir.mkdir(exist_ok=True)
            
            # Create profile CSV
            profile_data = []
            
            # Process personal information
            entities = analysis_results['entities']
            if entities['PERSON']:
                profile_data.append({
                    'section': 'personal',
                    'field': 'name',
                    'value': entities['PERSON'][0],
                    'confidence_score': 0.95,
                    'last_updated': datetime.now().isoformat()
                })
            
            # Process skills
            for skill in analysis_results['skills']:
                profile_data.append({
                    'section': 'skills',
                    'field': 'technical_skills',
                    'value': skill,
                    'confidence_score': 0.85,
                    'last_updated': datetime.now().isoformat()
                })
            
            # Process experience
            for exp in analysis_results['experience']:
                profile_data.append({
                    'section': 'experience',
                    'field': 'company',
                    'value': exp['organization'],
                    'confidence_score': 0.90,
                    'last_updated': datetime.now().isoformat()
                })
                profile_data.append({
                    'section': 'experience',
                    'field': 'duration',
                    'value': exp['dates'],
                    'confidence_score': 0.85,
                    'last_updated': datetime.now().isoformat()
                })
            
            # Process education
            for edu in analysis_results['education']:
                if edu['institution']:
                    profile_data.append({
                        'section': 'education',
                        'field': 'institution',
                        'value': edu['institution'],
                        'confidence_score': 0.90,
                        'last_updated': datetime.now().isoformat()
                    })
                if edu['date']:
                    profile_data.append({
                        'section': 'education',
                        'field': 'degree',
                        'value': edu['description'],
                        'confidence_score': 0.85,
                        'last_updated': datetime.now().isoformat()
                    })
            
            # Save to CSV
            profile_df = pd.DataFrame(profile_data)
            profile_df.to_csv(user_dir / "profile.csv", index=False)
            
            # Save raw analysis results
            with open(user_dir / "analysis_results.json", 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating profile: {str(e)}")
            return False
    
    def update_profile(self, user_id: str, updates: Dict) -> bool:
        """
        Update an existing profile with new information.
        
        Args:
            user_id: Unique identifier for the user
            updates: Dictionary containing updates
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            profile_path = self.profiles_dir / user_id / "profile.csv"
            if not profile_path.exists():
                logger.error(f"Profile not found for user: {user_id}")
                return False
            
            # Read existing profile
            profile_df = pd.read_csv(profile_path)
            
            # Apply updates
            for section, fields in updates.items():
                for field, value in fields.items():
                    mask = (profile_df['section'] == section) & (profile_df['field'] == field)
                    if mask.any():
                        profile_df.loc[mask, 'value'] = value
                        profile_df.loc[mask, 'last_updated'] = datetime.now().isoformat()
                    else:
                        # Add new entry if it doesn't exist
                        new_entry = {
                            'section': section,
                            'field': field,
                            'value': value,
                            'confidence_score': 1.0,
                            'last_updated': datetime.now().isoformat()
                        }
                        profile_df = pd.concat([profile_df, pd.DataFrame([new_entry])], ignore_index=True)
            
            # Save updated profile
            profile_df.to_csv(profile_path, index=False)
            return True
            
        except Exception as e:
            logger.error(f"Error updating profile: {str(e)}")
            return False
    
    def get_profile(self, user_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a user's profile.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Optional[pd.DataFrame]: Profile data if found, None otherwise
        """
        try:
            profile_path = self.profiles_dir / user_id / "profile.csv"
            if not profile_path.exists():
                logger.error(f"Profile not found for user: {user_id}")
                return None
            
            return pd.read_csv(profile_path)
            
        except Exception as e:
            logger.error(f"Error retrieving profile: {str(e)}")
            return None
