import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import seaborn as sns
import io
import base64

class ExploratoryAnalysis:
    """
    Module for creating interactive exploratory analyses
    """
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
        
    def create_text_length_analysis(self, text_data: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Create text length analysis charts
        """
        figures = {}
        
        # 1. Word length distribution (histograms)
        fig_hist = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribution - Words in title', 
                           'Distribution - Words in body',
                           'Distribution - Characters in title',
                           'Distribution - Characters in body'),
            vertical_spacing=0.3
        )
        
        # Title words histogram
        fig_hist.add_trace(
            go.Histogram(x=text_data['title_words'], name='Title words',
                        marker_color=self.colors[0], opacity=0.7),
            row=1, col=1
        )
        
        # Body words histogram
        fig_hist.add_trace(
            go.Histogram(x=text_data['body_words'], name='Body words',
                        marker_color='royalblue', opacity=0.9),
            row=1, col=2
        )
        
        # Title characters histogram
        fig_hist.add_trace(
            go.Histogram(x=text_data['title_chars'], name='Title characters',
                        marker_color=self.colors[2], opacity=0.7),
            row=2, col=1
        )
        
        # Body characters histogram
        fig_hist.add_trace(
            go.Histogram(x=text_data['body_chars'], name='Body characters',
                        marker_color=self.colors[3], opacity=0.7),
            row=2, col=2
        )
        
        fig_hist.update_layout(
            title="Text length distribution",
            showlegend=False,
            height=600,
            font=dict(size=12)
        )
        
        # Add axis labels
        fig_hist.update_xaxes(title_text="Number of words", row=1, col=1)
        fig_hist.update_xaxes(title_text="Number of words", row=1, col=2)
        fig_hist.update_xaxes(title_text="Number of characters", row=2, col=1)
        fig_hist.update_xaxes(title_text="Number of characters", row=2, col=2)
        
        figures['length_distribution'] = fig_hist
        
        return figures
    
    def create_word_frequency_analysis(self, word_freq_title: Dict[str, int], 
                                     word_freq_body: Dict[str, int]) -> Dict[str, go.Figure]:
        """
        Create word frequency analysis charts
        """
        figures = {}
        
        # 1. Top most frequent words - Horizontal bars
        top_words_title = dict(list(word_freq_title.items())[:20])
        top_words_body = dict(list(word_freq_body.items())[:20])
        
        fig_freq = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Most frequent words - Titles', 
                           'Most frequent words - Bodies'),
            horizontal_spacing=0.1
        )
        
        # Title bars
        fig_freq.add_trace(
            go.Bar(
                x=list(top_words_title.values()),
                y=list(top_words_title.keys()),
                orientation='h',
                name='Titles',
                marker_color=self.colors[0]
            ),
            row=1, col=1
        )
        
        # Body bars
        fig_freq.add_trace(
            go.Bar(
                x=list(top_words_body.values()),
                y=list(top_words_body.keys()),
                orientation='h',
                name='Bodies',
                marker_color=self.colors[1]
            ),
            row=1, col=2
        )
        
        fig_freq.update_layout(
            title="Most common words frequency",
            showlegend=False,
            height=600,
            font=dict(size=10)
        )
        
        fig_freq.update_xaxes(title_text="Frequency")
        
        figures['word_frequency'] = fig_freq
        
        return figures
    
    def create_tags_analysis(self, tags_data: Dict[str, any]) -> Dict[str, go.Figure]:
        """
        Create tags analysis charts
        """
        figures = {}
        
        # Top most popular tags
        top_tags = dict(list(tags_data['most_common_tags'].items())[:25])
        
        fig_top_tags = go.Figure(data=[
            go.Bar(
                x=list(top_tags.values()),
                y=list(top_tags.keys()),
                orientation='h',
                marker_color=self.colors[2]
            )
        ])
        
        fig_top_tags.update_layout(
            title="Top 25 most used tags",
            xaxis_title="Frequency",
            yaxis_title="Tags",
            height=700,
            font=dict(size=10)
        )
        
        figures['top_tags'] = fig_top_tags
        
        return figures
    
    def generate_wordcloud(self, word_freq: Dict[str, int], 
                         title: str = "Word Cloud") -> str:
        """
        Generate a word cloud and return the base64 encoded image
        """
        # Create the wordcloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)
        
        # Convert to base64 image
        img_buffer = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
