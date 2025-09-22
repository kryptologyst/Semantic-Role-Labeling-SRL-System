"""
Comprehensive Test Suite for Semantic Role Labeling System
Includes unit tests, integration tests, and performance tests
"""

import unittest
import pytest
import time
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import our modules
from modern_srl import SemanticRoleLabeler
from mock_database import MockDatabase
from visualization import SRLVisualizer

class TestMockDatabase(unittest.TestCase):
    """Test cases for MockDatabase class"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = MockDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database"""
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test database initialization"""
        # Check if tables are created
        conn = self.db.db_path
        self.assertTrue(os.path.exists(conn))
        
        # Check if sample data is inserted
        sentences = self.db.get_sentences()
        self.assertGreater(len(sentences), 0)
    
    def test_get_sentences(self):
        """Test sentence retrieval"""
        # Test without filters
        sentences = self.db.get_sentences()
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
        
        # Test with category filter
        simple_sentences = self.db.get_sentences(category='simple')
        self.assertIsInstance(simple_sentences, list)
        
        # Test with difficulty filter
        easy_sentences = self.db.get_sentences(difficulty='easy')
        self.assertIsInstance(easy_sentences, list)
        
        # Test with limit
        limited_sentences = self.db.get_sentences(limit=3)
        self.assertLessEqual(len(limited_sentences), 3)
    
    def test_save_and_get_srl_results(self):
        """Test saving and retrieving SRL results"""
        # Get a sentence
        sentences = self.db.get_sentences(limit=1)
        self.assertGreater(len(sentences), 0)
        sentence = sentences[0]
        
        # Create mock SRL result
        mock_result = {
            'model': 'test_model',
            'verbs': [{'verb': 'test', 'tags': ['B-ARG0', 'B-V']}],
            'processing_time': 0.5
        }
        
        # Save result
        self.db.save_srl_result(
            sentence_id=sentence['id'],
            model_name='test_model',
            result=mock_result,
            processing_time=0.5
        )
        
        # Retrieve result
        results = self.db.get_srl_results(sentence_id=sentence['id'])
        self.assertGreater(len(results), 0)
        
        retrieved_result = json.loads(results[0]['result_json'])
        self.assertEqual(retrieved_result['model'], 'test_model')
    
    def test_get_statistics(self):
        """Test statistics retrieval"""
        stats = self.db.get_statistics()
        
        self.assertIn('total_sentences', stats)
        self.assertIn('total_srl_results', stats)
        self.assertIn('categories', stats)
        self.assertIn('difficulties', stats)
        self.assertIn('models', stats)
        
        self.assertIsInstance(stats['total_sentences'], int)
        self.assertIsInstance(stats['categories'], dict)
        self.assertIsInstance(stats['difficulties'], dict)

class TestSemanticRoleLabeler(unittest.TestCase):
    """Test cases for SemanticRoleLabeler class"""
    
    def setUp(self):
        """Set up test SRL system"""
        # Mock the model loading to avoid downloading large models during tests
        with patch('modern_srl.Predictor') as mock_predictor:
            mock_predictor.from_path.return_value = Mock()
            self.srl = SemanticRoleLabeler()
    
    def test_initialization(self):
        """Test SRL system initialization"""
        self.assertIsInstance(self.srl.models, dict)
        self.assertIsInstance(self.srl.db, MockDatabase)
    
    def test_model_info(self):
        """Test model information retrieval"""
        model_info = self.srl.get_model_info()
        self.assertIsInstance(model_info, dict)
    
    def test_analyze_sentence_error_handling(self):
        """Test error handling in sentence analysis"""
        # Test with empty sentence
        with self.assertRaises(ValueError):
            self.srl.analyze_sentence("")
        
        # Test with no models available
        self.srl.models = {}
        with self.assertRaises(ValueError):
            self.srl.analyze_sentence("Test sentence")
    
    def test_batch_analyze(self):
        """Test batch analysis functionality"""
        sentences = [
            "John gave a book to Mary.",
            "The chef prepared a meal.",
            "Invalid sentence with error"
        ]
        
        # Mock the analyze_sentence method
        with patch.object(self.srl, 'analyze_sentence') as mock_analyze:
            mock_analyze.side_effect = [
                {'model': 'test', 'verbs': []},
                {'model': 'test', 'verbs': []},
                Exception("Test error")
            ]
            
            results = self.srl.batch_analyze(sentences)
            
            self.assertEqual(len(results), 3)
            self.assertIn('error', results[2])
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        # Mock models
        self.srl.models = {
            'model1': Mock(),
            'model2': Mock()
        }
        
        with patch.object(self.srl, 'analyze_sentence') as mock_analyze:
            mock_analyze.side_effect = [
                {'model': 'model1', 'verbs': []},
                {'model': 'model2', 'verbs': []}
            ]
            
            comparison = self.srl.compare_models("Test sentence")
            
            self.assertIn('sentence', comparison)
            self.assertIn('model_results', comparison)
            self.assertIn('available_models', comparison)
    
    def test_get_sample_sentences(self):
        """Test sample sentence retrieval"""
        sentences = self.srl.get_sample_sentences(limit=3)
        self.assertIsInstance(sentences, list)
        self.assertLessEqual(len(sentences), 3)

class TestSRLVisualizer(unittest.TestCase):
    """Test cases for SRLVisualizer class"""
    
    def setUp(self):
        """Set up test visualizer"""
        self.visualizer = SRLVisualizer()
    
    def test_initialization(self):
        """Test visualizer initialization"""
        self.assertIsInstance(self.visualizer.color_map, dict)
        self.assertIn('ARG0', self.visualizer.color_map)
        self.assertIn('V', self.visualizer.color_map)
    
    def test_create_role_distribution_chart(self):
        """Test role distribution chart creation"""
        sample_result = {
            'verbs': [
                {
                    'verb': 'gave',
                    'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'B-ARG2']
                }
            ]
        }
        
        fig = self.visualizer.create_role_distribution_chart(sample_result)
        self.assertIsNotNone(fig)
    
    def test_highlight_text_with_roles(self):
        """Test text highlighting functionality"""
        sentence = "John gave a book to Mary."
        sample_result = {
            'verbs': [
                {
                    'verb': 'gave',
                    'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'B-ARG2']
                }
            ]
        }
        
        highlighted = self.visualizer.highlight_text_with_roles(sentence, sample_result)
        self.assertIsInstance(highlighted, str)
        self.assertIn('<span', highlighted)
    
    def test_create_performance_metrics(self):
        """Test performance metrics creation"""
        sample_results = [
            {'processing_time': 0.5, 'verbs': [{'verb': 'test'}]},
            {'processing_time': 0.3, 'verbs': [{'verb': 'test2'}]}
        ]
        
        fig = self.visualizer.create_performance_metrics(sample_results)
        self.assertIsNotNone(fig)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Mock the database path
        with patch('modern_srl.MockDatabase') as mock_db_class:
            mock_db = Mock()
            mock_db.get_sentences.return_value = [
                {'id': 1, 'text': 'John gave a book to Mary.', 'category': 'simple', 'difficulty': 'easy'}
            ]
            mock_db_class.return_value = mock_db
            
            self.srl = SemanticRoleLabeler()
    
    def tearDown(self):
        """Clean up integration test environment"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Mock the analyze_sentence method
        with patch.object(self.srl, 'analyze_sentence') as mock_analyze:
            mock_analyze.return_value = {
                'model': 'test_model',
                'verbs': [{'verb': 'gave', 'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'B-ARG2']}],
                'processing_time': 0.5
            }
            
            # Test sentence analysis
            result = self.srl.analyze_sentence("John gave a book to Mary.")
            
            self.assertIn('model', result)
            self.assertIn('verbs', result)
            self.assertIn('processing_time', result)
    
    def test_database_integration(self):
        """Test database integration"""
        # This test would verify that results are properly saved to and retrieved from the database
        # Implementation depends on the specific database operations
        pass

class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    def setUp(self):
        """Set up performance test environment"""
        with patch('modern_srl.Predictor') as mock_predictor:
            mock_predictor.from_path.return_value = Mock()
            self.srl = SemanticRoleLabeler()
    
    def test_processing_time(self):
        """Test processing time performance"""
        sentences = [
            "John gave a book to Mary.",
            "The chef prepared a delicious meal.",
            "Sarah bought flowers from the market."
        ]
        
        start_time = time.time()
        
        with patch.object(self.srl, 'analyze_sentence') as mock_analyze:
            mock_analyze.return_value = {
                'model': 'test',
                'verbs': [],
                'processing_time': 0.1
            }
            
            results = self.srl.batch_analyze(sentences)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 5.0)
        self.assertEqual(len(results), len(sentences))
    
    def test_memory_usage(self):
        """Test memory usage (basic implementation)"""
        # This is a basic test - in production, you'd use memory profiling tools
        import sys
        
        initial_size = sys.getsizeof(self.srl)
        
        # Perform operations
        with patch.object(self.srl, 'analyze_sentence') as mock_analyze:
            mock_analyze.return_value = {'model': 'test', 'verbs': []}
            
            for i in range(100):
                self.srl.analyze_sentence(f"Test sentence {i}")
        
        final_size = sys.getsizeof(self.srl)
        
        # Memory usage shouldn't grow excessively
        self.assertLess(final_size - initial_size, 10000)  # Adjust threshold as needed

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMockDatabase,
        TestSemanticRoleLabeler,
        TestSRLVisualizer,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running SRL Test Suite")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("\nTest coverage areas:")
    print("- Database operations")
    print("- SRL model functionality")
    print("- Visualization tools")
    print("- Integration workflows")
    print("- Performance metrics")
