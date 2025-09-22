# Semantic Role Labeling (SRL) System

A modern, comprehensive implementation of Semantic Role Labeling using state-of-the-art NLP models and techniques. This project provides both programmatic APIs and an interactive web interface for analyzing semantic roles in text.

## Features

- **Multiple SRL Models**: Support for AllenNLP, Transformers, and custom implementations
- **Interactive Web UI**: Modern Streamlit-based interface for real-time analysis
- **Mock Database**: SQLite database with categorized sample sentences
- **Advanced Visualizations**: Dependency trees, role highlighting, and performance metrics
- **Batch Processing**: Analyze multiple sentences efficiently
- **Model Comparison**: Compare results across different SRL models
- **Comprehensive Testing**: Unit tests, integration tests, and performance benchmarks
- **Modern Architecture**: Clean, modular code with proper error handling

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/semantic-role-labeling.git
   cd semantic-role-labeling
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   streamlit run app.py
   ```

4. **Or use the Python API**
   ```python
   from modern_srl import SemanticRoleLabeler
   
   srl = SemanticRoleLabeler()
   result = srl.analyze_sentence("John gave a book to Mary on her birthday.")
   print(result)
   ```

## 📁 Project Structure

```
semantic-role-labeling/
├── app.py                 # Streamlit web application
├── modern_srl.py          # Core SRL implementation
├── mock_database.py       # Database management
├── visualization.py       # Advanced visualization tools
├── test_srl.py           # Comprehensive test suite
├── 0108.py              # Original implementation
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
└── srl_database.db      # SQLite database (created on first run)
```

## 🔧 API Reference

### SemanticRoleLabeler Class

The main class for performing semantic role labeling.

#### Methods

- `analyze_sentence(sentence: str, model_name: str = None) -> Dict[str, Any]`
  - Analyze a single sentence for semantic roles
  - Returns detailed SRL results with verbs, tags, and processing time

- `batch_analyze(sentences: List[str], model_name: str = None) -> List[Dict[str, Any]]`
  - Analyze multiple sentences in batch
  - Returns list of results with error handling

- `compare_models(sentence: str) -> Dict[str, Any]`
  - Compare results from all available models
  - Useful for model evaluation and selection

- `get_model_info() -> Dict[str, Any]`
  - Get information about available models
  - Returns model types, descriptions, and status

### MockDatabase Class

Database management for storing sentences and SRL results.

#### Methods

- `get_sentences(category: str = None, difficulty: str = None, limit: int = None) -> List[Dict[str, Any]]`
  - Retrieve sentences with optional filtering
  - Categories: simple, complex, modal, political, creative, medical, technical, legal, agricultural, aviation, scientific, sports
  - Difficulties: easy, medium, hard

- `save_srl_result(sentence_id: int, model_name: str, result: Dict[str, Any], processing_time: float)`
  - Save SRL analysis result to database

- `get_statistics() -> Dict[str, Any]`
  - Get database statistics and metrics

### SRLVisualizer Class

Advanced visualization tools for SRL results.

#### Methods

- `create_dependency_tree(sentence: str, srl_result: Dict[str, Any]) -> go.Figure`
  - Create interactive dependency tree visualization

- `create_role_distribution_chart(srl_result: Dict[str, Any]) -> go.Figure`
  - Create pie chart showing role distribution

- `create_performance_metrics(srl_results: List[Dict[str, Any]]) -> go.Figure`
  - Create gauge charts for performance metrics

- `highlight_text_with_roles(sentence: str, srl_result: Dict[str, Any]) -> str`
  - Create HTML with highlighted semantic roles

## Web Interface

The Streamlit web application provides:

- **Single Sentence Analysis**: Interactive text input with real-time analysis
- **Sample Sentences**: Browse categorized sentences from the database
- **Statistics Dashboard**: View database metrics and analysis history
- **Batch Processing**: Upload files or enter multiple sentences for batch analysis
- **Model Selection**: Choose between different SRL models
- **Visualizations**: Interactive charts and role highlighting

### Usage

1. Start the web app: `streamlit run app.py`
2. Open your browser to `http://localhost:8501`
3. Use the sidebar to select models and configure options
4. Analyze sentences using the main interface tabs

## Testing

Run the comprehensive test suite:

```bash
python test_srl.py
```

Or run specific test categories:

```bash
# Unit tests only
python -m unittest test_srl.TestMockDatabase -v

# Integration tests
python -m unittest test_srl.TestIntegration -v

# Performance tests
python -m unittest test_srl.TestPerformance -v
```

### Test Coverage

- ✅ Database operations (CRUD, filtering, statistics)
- ✅ SRL model functionality (analysis, batch processing, model comparison)
- ✅ Visualization tools (charts, highlighting, metrics)
- ✅ Integration workflows (end-to-end testing)
- ✅ Performance metrics (processing time, memory usage)

## Semantic Role Types

The system identifies various semantic roles:

- **ARG0**: Agent/Subject (who performs the action)
- **ARG1**: Theme/Object (what is affected by the action)
- **ARG2**: Goal/Recipient (to whom/where the action is directed)
- **ARG3**: Beneficiary (who benefits from the action)
- **ARG4**: Instrument (what is used to perform the action)
- **ARG5**: Attribute (properties or characteristics)
- **ARGM-TMP**: Temporal (when the action occurs)
- **ARGM-LOC**: Location (where the action occurs)
- **ARGM-MNR**: Manner (how the action is performed)
- **ARGM-CAU**: Cause (why the action occurs)
- **ARGM-PRP**: Purpose (for what reason the action occurs)
- **V**: Verb (the main action)

## Model Information

### Supported Models

1. **AllenNLP SRL Model**
   - BERT-based pretrained model
   - High accuracy for English text
   - Requires internet connection for initial download

2. **Transformers Models**
   - Various BERT and RoBERTa variants
   - Configurable model selection
   - Local processing capabilities

### Model Comparison

Use the `compare_models()` method to evaluate different models:

```python
comparison = srl.compare_models("John gave a book to Mary.")
for model_name, result in comparison['model_results'].items():
    print(f"{model_name}: {result['processing_time']:.3f}s")
```

## Performance Metrics

The system tracks various performance metrics:

- **Processing Time**: Time taken for analysis
- **Verb Count**: Number of verbs identified
- **Role Distribution**: Frequency of different semantic roles
- **Model Accuracy**: Comparison across different models
- **Memory Usage**: Resource utilization tracking

## 🛠️ Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# Database settings
DATABASE_PATH=srl_database.db

# Model settings
DEFAULT_MODEL=allennlp
CACHE_MODELS=true

# Performance settings
BATCH_SIZE=10
MAX_PROCESSING_TIME=30.0
```

### Custom Models

To add custom models, extend the `SemanticRoleLabeler` class:

```python
class CustomSRL(SemanticRoleLabeler):
    def _load_custom_model(self):
        # Load your custom model
        pass
    
    def analyze_sentence_custom(self, sentence: str):
        # Implement custom analysis
        pass
```

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python test_srl.py`
5. Commit your changes: `git commit -m "Add feature"`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run code formatting
black *.py

# Run linting
flake8 *.py

# Run tests with coverage
pytest test_srl.py --cov=modern_srl --cov-report=html
```

## Examples

### Basic Usage

```python
from modern_srl import SemanticRoleLabeler

# Initialize SRL system
srl = SemanticRoleLabeler()

# Analyze a sentence
sentence = "The chef prepared a delicious meal for the guests."
result = srl.analyze_sentence(sentence)

print(f"Processing time: {result['processing_time']:.3f}s")
for verb in result['verbs']:
    print(f"Verb: {verb['verb']}")
    print(f"Tags: {' '.join(verb['tags'])}")
```

### Batch Processing

```python
sentences = [
    "John gave a book to Mary.",
    "The teacher explained the concept clearly.",
    "Sarah bought flowers from the market."
]

results = srl.batch_analyze(sentences)
for i, result in enumerate(results):
    print(f"Sentence {i+1}: {len(result['verbs'])} verbs found")
```

### Database Operations

```python
from mock_database import MockDatabase

db = MockDatabase()

# Get sample sentences
sentences = db.get_sentences(category='simple', difficulty='easy', limit=5)
for sentence in sentences:
    print(f"{sentence['text']} ({sentence['category']})")

# Get statistics
stats = db.get_statistics()
print(f"Total sentences: {stats['total_sentences']}")
```

### Visualization

```python
from visualization import SRLVisualizer

visualizer = SRLVisualizer()

# Create role distribution chart
fig = visualizer.create_role_distribution_chart(result)
fig.show()

# Highlight text with roles
highlighted = visualizer.highlight_text_with_roles(sentence, result)
print(highlighted)
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure internet connection for initial model download
   - Check available disk space (models can be large)
   - Verify Python version compatibility

2. **Database Errors**
   - Ensure write permissions in the project directory
   - Check if database file is corrupted
   - Try deleting `srl_database.db` to recreate

3. **Memory Issues**
   - Reduce batch size for large datasets
   - Use smaller models for resource-constrained environments
   - Monitor memory usage with system tools

4. **Performance Issues**
   - Use GPU acceleration if available
   - Enable model caching
   - Consider using lighter models for real-time applications

### Getting Help

- Check the [Issues](https://github.com/kryptologyst/semantic-role-labeling/issues) page
- Create a new issue with detailed error information
- Include system information and error logs

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AllenNLP team for the SRL models and framework
- Hugging Face for the Transformers library
- Streamlit team for the web framework
- The NLP research community for semantic role labeling techniques

## References

1. Palmer, M., Gildea, D., & Xue, N. (2010). Semantic role labeling. Synthesis lectures on human language technologies, 3(1), 1-103.
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.
3. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.

## Future Enhancements

- [ ] Support for more languages
- [ ] Real-time streaming analysis
- [ ] Advanced model fine-tuning
- [ ] Integration with popular NLP pipelines
- [ ] Mobile app development
- [ ] Cloud deployment options
- [ ] Advanced visualization features
- [ ] Performance optimization


# Semantic-Role-Labeling-SRL-System
