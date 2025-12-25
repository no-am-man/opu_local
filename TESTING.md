# Testing Guide for OPU v3.4.3

## Overview

The OPU project now includes a comprehensive test suite targeting **100% code coverage** across all core modules.

## Quick Start

1. **Install test dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run all tests:**
   ```bash
   pytest tests/ -v
   ```

3. **Run with coverage:**
   ```bash
   pytest tests/ --cov=core --cov=utils --cov=main --cov-report=html
   ```

4. **View coverage report:**
   Open `htmlcov/index.html` in your browser.

## Test Files

| File | Module Tested | Test Count |
|------|---------------|------------|
| `test_genesis.py` | `core/genesis.py` | 19 tests |
| `test_perception.py` | `core/perception.py` | 14 tests |
| `test_cortex.py` | `core/cortex.py` | 27 tests |
| `test_expression.py` | `core/expression.py` | 25 tests |
| `test_persistence.py` | `utils/persistence.py` | 20 tests |
| `test_visualization.py` | `utils/visualization.py` | 17 tests |

**Total: 122+ tests covering all code paths**

## Coverage Requirements

The test suite is configured to **fail if coverage drops below 100%**. This ensures:

- All code paths are tested
- Edge cases are covered
- Error handling is verified
- Regression prevention

## Test Configuration

Configuration is in `pytest.ini`:

- **Coverage targets**: `core/`, `utils/`, `main.py`
- **Coverage threshold**: 100%
- **Test timeout**: 30 seconds per test
- **Output**: Verbose with missing line reports

## Running Specific Tests

```bash
# Run a specific test file
pytest tests/test_genesis.py -v

# Run a specific test
pytest tests/test_genesis.py::TestGenesisKernel::test_ethical_veto -v

# Run tests matching a pattern
pytest tests/ -k "veto" -v

# Run with markers
pytest tests/ -m unit -v
```

## Test Categories

- **Unit Tests**: Test individual functions/classes in isolation
- **Integration Tests**: Test component interactions
- **Edge Cases**: Test boundary conditions and error handling
- **Regression Tests**: Ensure bugs don't reappear

## Continuous Integration

The test suite is CI-ready:

- Exit code 1 on test failure
- Exit code 1 on coverage below 100%
- Generates XML reports for CI tools
- No external dependencies (mocks audio hardware)

## Key Test Features

### Fixtures

Reusable test fixtures in `conftest.py`:
- Sample audio vectors
- Temporary state files
- Mock audio streams
- Test data generators

### Mocks

External dependencies are mocked:
- `sounddevice` (audio I/O)
- File system operations
- Matplotlib display

### NumPy 2.0 Compatibility

All tests verify NumPy 2.0 compatibility:
- No deprecated types (`np.float_`, `np.int_`)
- Proper type conversion
- JSON serialization

## Example: Adding a New Test

```python
def test_new_feature():
    """Test description."""
    # Arrange
    component = Component()
    
    # Act
    result = component.new_method()
    
    # Assert
    assert result == expected_value
    assert len(component.history) > 0
```

## Troubleshooting

### Tests Fail with Import Errors

```bash
# Ensure you're in the project root
cd /path/to/opu_local

# Install dependencies
pip install -r requirements.txt
```

### Coverage Below 100%

Check the HTML report to see which lines are missing:
```bash
pytest tests/ --cov=core --cov-report=html
open htmlcov/index.html
```

### Audio Tests Fail

Audio tests use mocks and shouldn't require hardware. If they fail:
- Check that `pytest-mock` is installed
- Verify mocks are properly configured in `conftest.py`

## Best Practices

1. **Run tests before committing**
2. **Maintain 100% coverage**
3. **Test edge cases** (None, empty, boundaries)
4. **Use descriptive test names**
5. **Keep tests fast** (< 1 second each)
6. **Mock external dependencies**

## Next Steps

- Run the full test suite: `pytest tests/ -v`
- Check coverage: `pytest tests/ --cov --cov-report=html`
- Review test output for any failures
- Add tests for any new features

---

**The test suite ensures OPU v3.4.3 is robust, reliable, and maintainable.**

