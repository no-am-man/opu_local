# OPU v3.4.3 Test Suite

Comprehensive test suite with 100% code coverage target for the Orthogonal Processing Unit.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_genesis.py          # Safety Kernel tests
├── test_perception.py       # Perception tests
├── test_cortex.py           # Brain/Memory tests
├── test_expression.py       # Voice/Phoneme tests
├── test_persistence.py      # State persistence tests
├── test_visualization.py    # Visualization tests
└── run_tests.sh            # Test runner script
```

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
# Using pytest directly
pytest tests/ -v

# Using the test runner script
./tests/run_tests.sh

# With coverage report
pytest tests/ --cov=core --cov=utils --cov=main --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/test_genesis.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=core --cov=utils --cov=main --cov-report=term-missing --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`.

## Test Coverage

The test suite targets 100% code coverage for:

- **core/genesis.py**: Safety Kernel and ethical veto
- **core/perception.py**: Scale-invariant perception
- **core/cortex.py**: Introspection, memory abstraction, character evolution
- **core/expression.py**: Aesthetic feedback loop and phoneme analysis
- **utils/persistence.py**: State save/load with NumPy 2.0 compatibility
- **utils/visualization.py**: Cognitive map visualization

## Test Markers

Tests are organized with markers:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.slow`: Slow running tests
- `@pytest.mark.audio`: Tests requiring audio hardware

## Fixtures

Shared fixtures in `conftest.py`:

- `sample_audio_vector`: Sample audio input
- `sample_genomic_bit`: Sample genomic bit value
- `temp_state_file`: Temporary state file for persistence tests
- `mock_audio_stream`: Mock audio stream (avoids hardware dependencies)

## Continuous Integration

The test suite is configured to:

- Fail if coverage drops below 100%
- Timeout after 30 seconds per test
- Generate HTML coverage reports
- Support CI/CD pipelines

## Writing New Tests

When adding new features:

1. Add tests to the appropriate test file
2. Ensure 100% coverage is maintained
3. Use fixtures from `conftest.py` when possible
4. Mock external dependencies (audio, file I/O)
5. Test edge cases and error conditions

## Example Test

```python
def test_feature_name():
    """Test description."""
    # Arrange
    component = Component()
    
    # Act
    result = component.method()
    
    # Assert
    assert result == expected_value
```

