from typing import Dict, List, Optional
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import logging
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPAnalyzer:
    """Handles NLP analysis of resume text."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
            self.classifier = pipeline("zero-shot-classification")
            self.ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values
        """
        doc = self.nlp(text)
        entities = {
            'PERSON': [],
            'ORG': [],
            'DATE': [],
            'GPE': [],
            'SKILL': []
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using NER and custom patterns.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of extracted skills
        """
        # Common technical skills to look for
        common_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'sql'],
            'frameworks': ['django', 'flask', 'react', 'angular', 'vue'],
            'tools': ['git', 'docker', 'kubernetes', 'aws', 'jenkins']
        }
        
        skills = []
        text_lower = text.lower()
        
        # Extract skills using NER
        ner_results = self.ner(text)
        for result in ner_results:
            if result['entity'] == 'MISC' and result['score'] > 0.7:
                skills.append(result['word'])
        
        # Extract skills using common patterns
        for category, skill_list in common_skills.items():
            for skill in skill_list:
                if skill in text_lower:
                    skills.append(skill)
        
        return list(set(skills))
    
    def extract_experience(self, text: str) -> List[Dict]:
        """
        Extract work experience from text.
        
        Args:
            text: Input text
            
        Returns:
            List[Dict]: List of work experiences
        """
        experiences = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            doc = self.nlp(sentence)
            
            # Look for date patterns
            dates = [ent.text for ent in doc.ents if ent.label_ == 'DATE']
            
            # Look for organization names
            orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            
            if dates and orgs:
                experience = {
                    'organization': orgs[0],
                    'dates': dates[0],
                    'description': sentence
                }
                experiences.append(experience)
        
        return experiences
    
    def extract_education(self, text: str) -> List[Dict]:
        """
        Extract education information from text.
        
        Args:
            text: Input text
            
        Returns:
            List[Dict]: List of education entries
        """
        education = []
        sentences = sent_tokenize(text)
        
        # Keywords indicating education
        edu_keywords = ['degree', 'bachelor', 'master', 'phd', 'diploma', 'university', 'college']
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in edu_keywords):
                doc = self.nlp(sentence)
                
                # Extract organization and date
                org = next((ent.text for ent in doc.ents if ent.label_ == 'ORG'), None)
                date = next((ent.text for ent in doc.ents if ent.label_ == 'DATE'), None)
                
                if org or date:
                    edu_entry = {
                        'institution': org,
                        'date': date,
                        'description': sentence
                    }
                    education.append(edu_entry)
        
        return education
    
    def analyze_resume(self, text: str) -> Dict:
        """
        Perform complete analysis of resume text.
        
        Args:
            text: Input text
            
        Returns:
            Dict: Dictionary containing all extracted information
        """
        return {
            'entities': self.extract_entities(text),
            'skills': self.extract_skills(text),
            'experience': self.extract_experience(text),
            'education': self.extract_education(text)
        }
