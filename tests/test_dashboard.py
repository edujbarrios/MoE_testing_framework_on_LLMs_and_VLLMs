import pytest
from utils.dashboard import ComplexityDashboard

def test_dashboard_initialization():
    """Test dashboard initialization."""
    dashboard = ComplexityDashboard()
    assert hasattr(dashboard, 'metrics')
    assert hasattr(dashboard, 'layout')

def test_update_metrics():
    """Test metrics updates."""
    dashboard = ComplexityDashboard()
    
    # Test text complexity update
    dashboard.update_metrics('text_complexity', 0.5)
    assert len(dashboard.metrics['text_complexity']) == 1
    assert dashboard.metrics['text_complexity'][0] == 0.5
    
    # Test expert assignment update
    dashboard.update_metrics('expert_assignment', 1)
    assert 1 in dashboard.metrics['expert_assignments']
    assert dashboard.metrics['expert_assignments'][1] == 1

def test_metrics_limit():
    """Test that metrics lists don't grow indefinitely."""
    dashboard = ComplexityDashboard()
    
    # Add more than 100 values
    for i in range(110):
        dashboard.update_metrics('text_complexity', i)
    
    # Should keep only the last 100 values
    assert len(dashboard.metrics['text_complexity']) == 100

def test_create_complexity_table():
    """Test complexity table creation."""
    dashboard = ComplexityDashboard()
    dashboard.update_metrics('text_complexity', 0.5)
    dashboard.update_metrics('image_complexity', 0.7)
    
    table = dashboard._create_complexity_table()
    assert table is not None
