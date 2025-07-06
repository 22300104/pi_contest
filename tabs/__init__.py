"""탭 모듈"""
from .overall_stats import render_overall_stats_tab
from .column_analysis import render_column_analysis_tab
from .data_preview import render_data_preview_tab

__all__ = [
    'render_overall_stats_tab',
    'render_column_analysis_tab', 
    'render_data_preview_tab'
]