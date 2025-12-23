"""
Tests for core/patterns/observer.py - Observer Pattern
100% code coverage target
"""

import pytest
from core.patterns.observer import OPUObserver, ObservableOPU


class MockObserver(OPUObserver):
    """Mock observer for testing."""
    
    def __init__(self):
        self.notified_states = []
    
    def on_state_changed(self, state):
        """Record notified state."""
        self.notified_states.append(state)


class TestOPUObserver:
    """Test suite for abstract OPUObserver class."""
    
    def test_observer_is_abstract(self):
        """Test that OPUObserver cannot be instantiated."""
        from abc import ABC
        assert issubclass(OPUObserver, ABC)
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            OPUObserver()


class TestObservableOPU:
    """Test suite for ObservableOPU mixin."""
    
    def test_init(self):
        """Test ObservableOPU initialization."""
        observable = ObservableOPU()
        assert len(observable._observers) == 0
    
    def test_attach_observer(self):
        """Test attaching an observer."""
        observable = ObservableOPU()
        observer = MockObserver()
        observable.attach_observer(observer)
        assert observer in observable._observers
        assert len(observable._observers) == 1
    
    def test_attach_observer_duplicate(self):
        """Test that duplicate observers are not added."""
        observable = ObservableOPU()
        observer = MockObserver()
        observable.attach_observer(observer)
        observable.attach_observer(observer)  # Try to add again
        assert len(observable._observers) == 1
    
    def test_detach_observer(self):
        """Test detaching an observer."""
        observable = ObservableOPU()
        observer = MockObserver()
        observable.attach_observer(observer)
        observable.detach_observer(observer)
        assert observer not in observable._observers
        assert len(observable._observers) == 0
    
    def test_detach_observer_not_attached(self):
        """Test detaching an observer that was never attached."""
        observable = ObservableOPU()
        observer = MockObserver()
        # Should not raise error
        observable.detach_observer(observer)
        assert len(observable._observers) == 0
    
    def test_notify_observers(self):
        """Test notifying observers."""
        observable = ObservableOPU()
        observer1 = MockObserver()
        observer2 = MockObserver()
        observable.attach_observer(observer1)
        observable.attach_observer(observer2)
        
        state = {'s_score': 1.5, 'maturity': 0.5}
        observable.notify_observers(state)
        
        assert len(observer1.notified_states) == 1
        assert len(observer2.notified_states) == 1
        assert observer1.notified_states[0] == state
        assert observer2.notified_states[0] == state
    
    def test_notify_observers_error_handling(self):
        """Test that observer errors don't break notification."""
        observable = ObservableOPU()
        
        class FailingObserver(OPUObserver):
            def on_state_changed(self, state):
                raise ValueError("Test error")
        
        class GoodObserver(OPUObserver):
            def __init__(self):
                self.notified = False
            def on_state_changed(self, state):
                self.notified = True
        
        failing = FailingObserver()
        good = GoodObserver()
        observable.attach_observer(failing)
        observable.attach_observer(good)
        
        # Should not raise, and good observer should still be notified
        observable.notify_observers({'test': 'state'})
        assert good.notified is True
    
    def test_notify_observers_empty(self):
        """Test notifying with no observers."""
        observable = ObservableOPU()
        # Should not raise error
        observable.notify_observers({'test': 'state'})

