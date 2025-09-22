# Contributing to Semantic Role Labeling

Thank you for your interest in contributing to this project! We welcome contributions from the community.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/semantic-role-labeling.git
   cd semantic-role-labeling
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🛠️ Development Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install development dependencies**:
   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Run tests** to ensure everything works:
   ```bash
   python test_srl.py
   ```

## 📝 Code Style

We follow Python best practices and use automated tools for code formatting:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

### Formatting Code

```bash
# Format code with Black
black *.py

# Check linting with Flake8
flake8 *.py

# Type checking with MyPy
mypy *.py
```

### Code Guidelines

- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write tests for new functionality
- Keep functions focused and small
- Follow PEP 8 style guidelines

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python test_srl.py

# Run specific test classes
python -m unittest test_srl.TestMockDatabase -v

# Run with coverage
pytest test_srl.py --cov=modern_srl --cov-report=html
```

### Writing Tests

- Write tests for all new functionality
- Include both positive and negative test cases
- Test edge cases and error conditions
- Aim for high test coverage
- Use descriptive test names

### Test Structure

```python
class TestYourFeature(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        pass
    
    def test_feature_positive_case(self):
        """Test feature works correctly"""
        # Test implementation
        pass
    
    def test_feature_error_handling(self):
        """Test feature handles errors gracefully"""
        # Error handling test
        pass
```

## 📋 Pull Request Process

1. **Ensure your code is properly formatted**:
   ```bash
   black *.py
   flake8 *.py
   ```

2. **Run all tests** and ensure they pass:
   ```bash
   python test_srl.py
   ```

3. **Update documentation** if you've added new features or changed APIs

4. **Commit your changes** with descriptive messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Screenshots for UI changes
   - Test results

## 🐛 Bug Reports

When reporting bugs, please include:

- **Description** of the bug
- **Steps to reproduce** the issue
- **Expected behavior** vs actual behavior
- **System information** (OS, Python version, etc.)
- **Error messages** and logs
- **Screenshots** if applicable

## ✨ Feature Requests

When requesting features, please include:

- **Clear description** of the feature
- **Use case** and motivation
- **Proposed implementation** (if you have ideas)
- **Alternative solutions** considered
- **Additional context** or examples

## 📚 Documentation

### Code Documentation

- Add docstrings to all functions and classes
- Include parameter descriptions and return values
- Provide usage examples in docstrings
- Update README.md for significant changes

### API Documentation

- Document new API endpoints
- Include request/response examples
- Update type hints and annotations
- Maintain consistency with existing patterns

## 🔍 Code Review Process

### For Contributors

- Respond to review feedback promptly
- Make requested changes clearly
- Test changes thoroughly
- Keep commits focused and atomic
- Update documentation as needed

### For Reviewers

- Be constructive and helpful
- Focus on code quality and functionality
- Check for security issues
- Verify tests and documentation
- Approve when ready

## 🏷️ Issue Labels

We use the following labels for issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information is requested
- `wontfix`: Will not be worked on

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: For security issues or private matters

## 🎯 Areas for Contribution

We especially welcome contributions in these areas:

- **New SRL Models**: Adding support for additional models
- **Language Support**: Extending to other languages
- **Performance**: Optimizing processing speed and memory usage
- **Visualization**: Improving charts and interactive features
- **Documentation**: Improving guides and examples
- **Testing**: Adding more comprehensive test coverage
- **UI/UX**: Enhancing the web interface
- **API**: Adding new endpoints and functionality

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## 🙏 Recognition

Contributors will be recognized in:

- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation
- GitHub contributor statistics

Thank you for contributing to the Semantic Role Labeling project! 🎉
