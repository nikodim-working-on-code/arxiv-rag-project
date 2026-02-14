# config management
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    
    #  groq api - what i used, can be any
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
    
    # can be any
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-ai/nomic-embed-text-v1.5')
    
    # default weights
    DEFAULT_WEIGHTS = {
        'alpha': 0.4,
        'beta': 0.25,
        'gamma': 0.25,
        'delta': 0.1,
        'tau': 0.5,
        'lambda': 0.01,
        'num_queries': 0,
        'avg_loss': None
    }
    
    WEIGHTS_FILE = 'weights.json'
    HISTORY_FILE = 'optimization_history.json'
    TEMP_PDF_DIR = './temp_pdfs'
    
    # chunking parameters - better check what suits u more
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80
    
    @classmethod
    def validate(cls):
        """Validate that required API keys are set"""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found! Create .env file based on .env.example"
            )
        return True


Config.validate()

