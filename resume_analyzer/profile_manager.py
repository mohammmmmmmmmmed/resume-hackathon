import csv
import json
from typing import Dict, List, Optional
import os
from datetime import datetime
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfileManager:
    """Manages profile storage and retrieval."""
    
    def __init__(self):
        self.profiles_dir = "data/profiles"
        os.makedirs(self.profiles_dir, exist_ok=True)
    
    def _get_profile_path(self, user_id: str) -> str:
        """Get the file path for a profile."""
        return os.path.join(self.profiles_dir, f"{user_id}.json")
    
    def save_profile(self, profile_data: Dict, user_id: Optional[str] = None) -> str:
        """
        Save a profile to disk.
        
        Args:
            profile_data: Profile data to save
            user_id: Optional user ID. If not provided, a new one will be generated.
            
        Returns:
            str: User ID of the saved profile
        """
        try:
            if not user_id:
                user_id = str(uuid.uuid4())
            
            profile_data['user_id'] = user_id
            profile_data['last_updated'] = datetime.now().isoformat()
            
            profile_path = self._get_profile_path(user_id)
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            return user_id
        except Exception as e:
            logger.error(f"Error saving profile: {str(e)}")
            raise
    
    def get_profile(self, user_id: str) -> Optional[Dict]:
        """
        Retrieve a profile by user ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            Dict: Profile data if found, None otherwise
        """
        try:
            profile_path = self._get_profile_path(user_id)
            if not os.path.exists(profile_path):
                return None
            
            with open(profile_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error retrieving profile: {str(e)}")
            return None
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete a profile by user ID.
        
        Args:
            user_id: User ID to delete
            
        Returns:
            bool: True if deleted, False otherwise
        """
        try:
            profile_path = self._get_profile_path(user_id)
            if os.path.exists(profile_path):
                os.remove(profile_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting profile: {str(e)}")
            return False
    
    def get_all_profiles(self) -> List[Dict]:
        """
        Retrieve all profiles.
        
        Returns:
            List[Dict]: List of all profiles
        """
        try:
            profiles = []
            for filename in os.listdir(self.profiles_dir):
                if filename.endswith('.json'):
                    user_id = filename[:-5]  # Remove .json extension
                    profile = self.get_profile(user_id)
                    if profile:
                        profiles.append(profile)
            return profiles
        except Exception as e:
            logger.error(f"Error retrieving all profiles: {str(e)}")
            return [] 