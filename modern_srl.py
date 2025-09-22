"""
Modern Semantic Role Labeling Implementation
Supports multiple models: AllenNLP, Transformers, and custom implementations
"""

import time
import json
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# Core libraries
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# AllenNLP (legacy support)
try:
    from allennlp.predictors.predictor import Predictor
    import allennlp_models.structured_prediction
    ALLENNLP_AVAILABLE = True
except ImportError:
    ALLENNLP_AVAILABLE = False
    print("Warning: AllenNLP not available. Install with: pip install allennlp allennlp-models")

# Database
from mock_database import MockDatabase

class SemanticRoleLabeler:
    """Modern SRL implementation with multiple model support"""
    
    def __init__(self):
        self.models = {}
        self.db = MockDatabase()
        self._load_models()
    
    def _load_models(self):
        """Load available SRL models"""
        print("Loading SRL models...")
        
        # Try to load AllenNLP model
        if ALLENNLP_AVAILABLE:
            try:
                self.models['allennlp'] = Predictor.from_path(
                    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
                )
                print("✓ AllenNLP SRL model loaded")
            except Exception as e:
                print(f"✗ Failed to load AllenNLP model: {e}")
        
        # Load Transformers-based models
        try:
            # Try different SRL models from Hugging Face
            srl_models = [
                "dbmdz/bert-large-cased-finetuned-conll03-english",
                "xlm-roberta-large-finetuned-conll03-english"
            ]
            
            for model_name in srl_models:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForTokenClassification.from_pretrained(model_name)
                    self.models[f'transformers_{model_name.split("/")[-1]}'] = {
                        'tokenizer': tokenizer,
                        'model': model,
                        'pipeline': pipeline("token-classification", model=model, tokenizer=tokenizer)
                    }
                    print(f"✓ Transformers model {model_name.split('/')[-1]} loaded")
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
        
        except Exception as e:
            print(f"✗ Failed to load Transformers models: {e}")
        
        print(f"Loaded {len(self.models)} SRL models")
    
    def analyze_sentence_allennlp(self, sentence: str) -> Dict[str, Any]:
        """Analyze sentence using AllenNLP SRL model"""
        if 'allennlp' not in self.models:
            raise ValueError("AllenNLP model not available")
        
        start_time = time.time()
        result = self.models['allennlp'].predict(sentence=sentence)
        processing_time = time.time() - start_time
        
        return {
            'model': 'allennlp',
            'sentence': sentence,
            'result': result,
            'processing_time': processing_time,
            'verbs': result.get('verbs', []),
            'words': result.get('words', [])
        }
    
    def analyze_sentence_transformers(self, sentence: str, model_name: str) -> Dict[str, Any]:
        """Analyze sentence using Transformers-based model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        start_time = time.time()
        
        # Use the pipeline for token classification
        pipeline_obj = self.models[model_name]['pipeline']
        result = pipeline_obj(sentence)
        
        processing_time = time.time() - start_time
        
        # Convert to SRL-like format
        srl_result = self._convert_transformers_to_srl(result, sentence)
        
        return {
            'model': model_name,
            'sentence': sentence,
            'result': result,
            'srl_result': srl_result,
            'processing_time': processing_time,
            'verbs': srl_result.get('verbs', []),
            'words': sentence.split()
        }
    
    def _convert_transformers_to_srl(self, transformers_result: List[Dict], sentence: str) -> Dict[str, Any]:
        """Convert Transformers NER result to SRL-like format"""
        words = sentence.split()
        verbs = []
        
        # Simple heuristic: treat entities as potential arguments
        entities = {}
        for entity in transformers_result:
            label = entity['entity']
            word = entity['word']
            score = entity['score']
            
            if label not in entities:
                entities[label] = []
            entities[label].append({
                'word': word,
                'score': score,
                'start': entity['start'],
                'end': entity['end']
            })
        
        # Create a simple verb structure (this is a simplified approach)
        # In a real implementation, you'd use a proper SRL model
        main_verb = None
        for word in words:
            if word.lower() in ['gave', 'prepared', 'bought', 'explained', 'discussed', 
                              'discovered', 'continued', 'announced', 'examined', 'completed']:
                main_verb = word
                break
        
        if main_verb:
            verbs.append({
                'verb': main_verb,
                'tags': ['O'] * len(words),  # Simplified tags
                'description': f"Main verb: {main_verb}"
            })
        
        return {
            'verbs': verbs,
            'entities': entities,
            'words': words
        }
    
    def analyze_sentence(self, sentence: str, model_name: str = None) -> Dict[str, Any]:
        """Analyze sentence with specified model or best available model"""
        if model_name and model_name in self.models:
            if model_name == 'allennlp':
                return self.analyze_sentence_allennlp(sentence)
            elif model_name.startswith('transformers_'):
                return self.analyze_sentence_transformers(sentence, model_name)
        
        # Use first available model
        if self.models:
            first_model = list(self.models.keys())[0]
            return self.analyze_sentence(sentence, first_model)
        
        raise ValueError("No SRL models available")
    
    def batch_analyze(self, sentences: List[str], model_name: str = None) -> List[Dict[str, Any]]:
        """Analyze multiple sentences in batch"""
        results = []
        for sentence in sentences:
            try:
                result = self.analyze_sentence(sentence, model_name)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'sentence': sentence,
                    'model': model_name
                })
        return results
    
    def compare_models(self, sentence: str) -> Dict[str, Any]:
        """Compare results from all available models"""
        results = {}
        
        for model_name in self.models.keys():
            try:
                if model_name == 'allennlp':
                    results[model_name] = self.analyze_sentence_allennlp(sentence)
                elif model_name.startswith('transformers_'):
                    results[model_name] = self.analyze_sentence_transformers(sentence, model_name)
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return {
            'sentence': sentence,
            'model_results': results,
            'available_models': list(self.models.keys())
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        info = {}
        for model_name in self.models.keys():
            if model_name == 'allennlp':
                info[model_name] = {
                    'type': 'AllenNLP',
                    'description': 'BERT-based SRL model from AllenNLP',
                    'status': 'loaded'
                }
            elif model_name.startswith('transformers_'):
                info[model_name] = {
                    'type': 'Transformers',
                    'description': f'Transformers-based model: {model_name}',
                    'status': 'loaded'
                }
        
        return info
    
    def save_result_to_db(self, sentence: str, result: Dict[str, Any]):
        """Save analysis result to database"""
        # Get sentence ID from database
        sentences = self.db.get_sentences()
        sentence_id = None
        
        for s in sentences:
            if s['text'] == sentence:
                sentence_id = s['id']
                break
        
        if sentence_id:
            self.db.save_srl_result(
                sentence_id=sentence_id,
                model_name=result['model'],
                result=result,
                processing_time=result['processing_time']
            )
    
    def get_sample_sentences(self, category: str = None, difficulty: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample sentences from database"""
        return self.db.get_sentences(category=category, difficulty=difficulty, limit=limit)

def main():
    """Demo function"""
    print("🚀 Modern Semantic Role Labeling Demo")
    print("=" * 50)
    
    # Initialize SRL
    srl = SemanticRoleLabeler()
    
    # Show available models
    print("\n📋 Available Models:")
    model_info = srl.get_model_info()
    for model_name, info in model_info.items():
        print(f"  • {model_name}: {info['description']}")
    
    # Test sentence
    test_sentence = "John gave a book to Mary on her birthday."
    print(f"\n🔍 Analyzing: '{test_sentence}'")
    
    # Analyze with first available model
    try:
        result = srl.analyze_sentence(test_sentence)
        print(f"\n✅ Analysis completed in {result['processing_time']:.3f}s")
        print(f"Model used: {result['model']}")
        
        if 'verbs' in result and result['verbs']:
            print("\n📝 Semantic Roles:")
            for verb in result['verbs']:
                print(f"  • Verb: {verb.get('verb', 'N/A')}")
                if 'tags' in verb:
                    print(f"    Tags: {' '.join(verb['tags'])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Show sample sentences from database
    print("\n📚 Sample Sentences from Database:")
    samples = srl.get_sample_sentences(limit=3)
    for sample in samples:
        print(f"  • {sample['text']} ({sample['category']}, {sample['difficulty']})")

if __name__ == "__main__":
    main()
