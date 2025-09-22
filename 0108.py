"""
Project 108: Modern Semantic Role Labeling
==========================================

Semantic Role Labeling (SRL) is the process of assigning roles to words or phrases 
in a sentence to determine who did what to whom, when, and how. It helps in 
understanding sentence meaning beyond grammar.

This modern implementation includes:
- Multiple SRL models (AllenNLP, Transformers)
- Interactive web interface
- Mock database with sample sentences
- Advanced visualizations
- Batch processing capabilities
- Comprehensive testing

Usage:
    python 0108.py                    # Run basic demo
    streamlit run app.py             # Launch web interface
    python modern_srl.py             # Use programmatic API
    python test_srl.py               # Run test suite
"""

import warnings
warnings.filterwarnings("ignore")

# Modern implementation with error handling
def run_basic_demo():
    """Run basic SRL demo with modern error handling"""
    print("🧠 Semantic Role Labeling Demo")
    print("=" * 50)
    
    try:
        # Try to import AllenNLP
        from allennlp.predictors.predictor import Predictor
        import allennlp_models.structured_prediction
        
        print("Loading AllenNLP SRL model...")
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
        )
        
        # Sample sentences
        sentences = [
            "John gave a book to Mary on her birthday.",
            "The chef prepared a delicious meal for the guests.",
            "Sarah bought flowers from the market yesterday."
        ]
        
        for i, sentence in enumerate(sentences, 1):
            print(f"\n📝 Sentence {i}: {sentence}")
            
            # Perform Semantic Role Labeling
            result = predictor.predict(sentence=sentence)
            
            print("🔍 Semantic Roles:")
            for verb in result['verbs']:
                print(f"  ▶ Verb: {verb['verb']}")
                print(f"     Tags: {' '.join(verb['tags'])}")
                
                # Explain the roles
                words = result['words']
                tags = verb['tags']
                
                print("     Analysis:")
                for j, (word, tag) in enumerate(zip(words, tags)):
                    if tag != 'O':
                        role = tag.split('-')[-1] if '-' in tag else tag
                        print(f"       {word} → {role}")
        
        print("\n✅ Demo completed successfully!")
        print("\nFor more features, try:")
        print("  • streamlit run app.py (Web interface)")
        print("  • python modern_srl.py (Advanced API)")
        print("  • python test_srl.py (Run tests)")
        
    except ImportError as e:
        print("❌ AllenNLP not available. Install with:")
        print("   pip install allennlp allennlp-models")
        print(f"   Error: {e}")
        
        print("\n🔄 Trying alternative implementation...")
        try:
            from modern_srl import SemanticRoleLabeler
            srl = SemanticRoleLabeler()
            
            sentence = "John gave a book to Mary on her birthday."
            result = srl.analyze_sentence(sentence)
            
            print(f"\n📝 Sentence: {sentence}")
            print("🔍 Semantic Roles:")
            for verb in result['verbs']:
                print(f"  ▶ Verb: {verb.get('verb', 'N/A')}")
                if 'tags' in verb:
                    print(f"     Tags: {' '.join(verb['tags'])}")
            
            print("\n✅ Alternative implementation successful!")
            
        except Exception as e2:
            print(f"❌ Alternative implementation failed: {e2}")
            print("\nPlease install dependencies:")
            print("   pip install -r requirements.txt")
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection (model download required)")
        print("2. Ensure sufficient disk space")
        print("3. Try: pip install --upgrade allennlp allennlp-models")

def run_batch_demo():
    """Run batch processing demo"""
    print("\n🔄 Batch Processing Demo")
    print("=" * 30)
    
    try:
        from modern_srl import SemanticRoleLabeler
        from mock_database import MockDatabase
        
        srl = SemanticRoleLabeler()
        db = MockDatabase()
        
        # Get sample sentences
        sentences_data = db.get_sentences(limit=5)
        sentences = [s['text'] for s in sentences_data]
        
        print(f"Processing {len(sentences)} sentences...")
        
        # Batch analysis
        results = srl.batch_analyze(sentences)
        
        successful = len([r for r in results if 'error' not in r])
        failed = len([r for r in results if 'error' in r])
        
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        # Show results
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"\nSentence {i+1}: {sentences[i]}")
                print(f"  Model: {result['model']}")
                print(f"  Processing time: {result['processing_time']:.3f}s")
                print(f"  Verbs found: {len(result['verbs'])}")
        
    except Exception as e:
        print(f"❌ Batch demo failed: {e}")

if __name__ == "__main__":
    # Run basic demo
    run_basic_demo()
    
    # Run batch demo if basic demo succeeded
    try:
        run_batch_demo()
    except:
        pass  # Skip if batch demo fails