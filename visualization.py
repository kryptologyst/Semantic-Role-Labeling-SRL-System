"""
Advanced Visualization Tools for Semantic Role Labeling
Includes dependency trees, role highlighting, and interactive charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from collections import Counter, defaultdict

class SRLVisualizer:
    """Advanced visualization tools for SRL results"""
    
    def __init__(self):
        self.color_map = {
            'ARG0': '#FF6B6B',  # Red - Agent/Subject
            'ARG1': '#4ECDC4',  # Teal - Theme/Object
            'ARG2': '#45B7D1',  # Blue - Goal/Recipient
            'ARG3': '#96CEB4',  # Green - Beneficiary
            'ARG4': '#FFEAA7',  # Yellow - Instrument
            'ARG5': '#DDA0DD',  # Plum - Attribute
            'ARGM-TMP': '#FFB347',  # Orange - Temporal
            'ARGM-LOC': '#87CEEB',  # Sky Blue - Location
            'ARGM-MNR': '#F0E68C',  # Khaki - Manner
            'ARGM-CAU': '#FFA07A',  # Light Salmon - Cause
            'ARGM-PRP': '#98FB98',  # Pale Green - Purpose
            'V': '#2C3E50',  # Dark Blue - Verb
            'O': '#BDC3C7'   # Gray - Other
        }
    
    def create_dependency_tree(self, sentence: str, srl_result: Dict[str, Any]) -> go.Figure:
        """Create an interactive dependency tree visualization"""
        words = sentence.split()
        verbs = srl_result.get('verbs', [])
        
        if not verbs:
            return go.Figure()
        
        # Create network graph
        G = nx.DiGraph()
        
        # Add nodes (words)
        for i, word in enumerate(words):
            G.add_node(i, label=word, pos=(i, 0))
        
        # Add edges based on SRL roles
        for verb in verbs:
            verb_word = verb.get('verb', '')
            tags = verb.get('tags', [])
            
            # Find verb position
            verb_pos = None
            for i, word in enumerate(words):
                if word.lower() == verb_word.lower():
                    verb_pos = i
                    break
            
            if verb_pos is not None:
                # Add edges from verb to arguments
                for i, tag in enumerate(tags):
                    if tag != 'O' and tag != 'V' and i < len(words):
                        role = tag.split('-')[-1] if '-' in tag else tag
                        G.add_edge(verb_pos, i, label=role, color=self.color_map.get(role, '#BDC3C7'))
        
        # Create plotly visualization
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [G.nodes[node]['label'] for node in G.nodes()]
        
        # Extract edge information
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            if 'label' in G.edges[edge]:
                edge_info.append(G.edges[edge]['label'])
        
        # Create traces
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=30, color='lightblue', line=dict(width=2, color='darkblue')),
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            hovertext=[f"Word: {word}" for word in node_text],
            showlegend=False
        ))
        
        fig.update_layout(
            title="Semantic Role Dependency Tree",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive dependency tree showing semantic roles",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="gray", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_role_distribution_chart(self, srl_result: Dict[str, Any]) -> go.Figure:
        """Create a chart showing distribution of semantic roles"""
        verbs = srl_result.get('verbs', [])
        
        if not verbs:
            return go.Figure()
        
        role_counts = Counter()
        
        for verb in verbs:
            tags = verb.get('tags', [])
            for tag in tags:
                if tag != 'O':
                    role = tag.split('-')[-1] if '-' in tag else tag
                    role_counts[role] += 1
        
        if not role_counts:
            return go.Figure()
        
        # Create pie chart
        fig = px.pie(
            values=list(role_counts.values()),
            names=list(role_counts.keys()),
            title="Semantic Role Distribution",
            color_discrete_sequence=[self.color_map.get(role, '#BDC3C7') for role in role_counts.keys()]
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def create_role_timeline(self, srl_results: List[Dict[str, Any]]) -> go.Figure:
        """Create a timeline showing role usage over multiple analyses"""
        if not srl_results:
            return go.Figure()
        
        timeline_data = []
        
        for i, result in enumerate(srl_results):
            verbs = result.get('verbs', [])
            for verb in verbs:
                tags = verb.get('tags', [])
                for tag in tags:
                    if tag != 'O':
                        role = tag.split('-')[-1] if '-' in tag else tag
                        timeline_data.append({
                            'analysis': i + 1,
                            'role': role,
                            'verb': verb.get('verb', 'Unknown')
                        })
        
        if not timeline_data:
            return go.Figure()
        
        df = pd.DataFrame(timeline_data)
        
        # Create stacked bar chart
        fig = px.bar(
            df,
            x='analysis',
            color='role',
            title="Semantic Role Usage Timeline",
            color_discrete_map=self.color_map
        )
        
        fig.update_layout(
            xaxis_title="Analysis Number",
            yaxis_title="Number of Roles",
            height=400
        )
        
        return fig
    
    def create_model_comparison(self, comparison_results: Dict[str, Any]) -> go.Figure:
        """Create a comparison chart for different models"""
        model_results = comparison_results.get('model_results', {})
        
        if not model_results:
            return go.Figure()
        
        # Extract processing times and verb counts
        models = []
        processing_times = []
        verb_counts = []
        
        for model_name, result in model_results.items():
            if 'error' not in result:
                models.append(model_name)
                processing_times.append(result.get('processing_time', 0))
                verb_counts.append(len(result.get('verbs', [])))
        
        if not models:
            return go.Figure()
        
        # Create subplot with two y-axes
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Processing Time", "Verb Count"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Processing time chart
        fig.add_trace(
            go.Bar(x=models, y=processing_times, name="Processing Time (s)", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Verb count chart
        fig.add_trace(
            go.Bar(x=models, y=verb_counts, name="Verb Count", marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_role_heatmap(self, srl_results: List[Dict[str, Any]]) -> go.Figure:
        """Create a heatmap showing role co-occurrence patterns"""
        if not srl_results:
            return go.Figure()
        
        role_cooccurrence = defaultdict(int)
        all_roles = set()
        
        for result in srl_results:
            verbs = result.get('verbs', [])
            sentence_roles = set()
            
            for verb in verbs:
                tags = verb.get('tags', [])
                for tag in tags:
                    if tag != 'O':
                        role = tag.split('-')[-1] if '-' in tag else tag
                        sentence_roles.add(role)
                        all_roles.add(role)
            
            # Count co-occurrences
            roles_list = list(sentence_roles)
            for i, role1 in enumerate(roles_list):
                for role2 in roles_list[i+1:]:
                    pair = tuple(sorted([role1, role2]))
                    role_cooccurrence[pair] += 1
        
        if not role_cooccurrence:
            return go.Figure()
        
        # Create matrix
        roles_list = sorted(list(all_roles))
        matrix = np.zeros((len(roles_list), len(roles_list)))
        
        for (role1, role2), count in role_cooccurrence.items():
            i1 = roles_list.index(role1)
            i2 = roles_list.index(role2)
            matrix[i1][i2] = count
            matrix[i2][i1] = count
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=roles_list,
            y=roles_list,
            colorscale='Blues',
            text=matrix.astype(int),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Semantic Role Co-occurrence Heatmap",
            xaxis_title="Roles",
            yaxis_title="Roles",
            height=500
        )
        
        return fig
    
    def highlight_text_with_roles(self, sentence: str, srl_result: Dict[str, Any]) -> str:
        """Create HTML with highlighted semantic roles"""
        words = sentence.split()
        verbs = srl_result.get('verbs', [])
        
        if not verbs:
            return sentence
        
        highlighted_words = []
        
        for i, word in enumerate(words):
            # Find the most specific role for this word
            best_role = 'O'
            best_color = '#BDC3C7'
            
            for verb in verbs:
                tags = verb.get('tags', [])
                if i < len(tags):
                    tag = tags[i]
                    if tag != 'O':
                        role = tag.split('-')[-1] if '-' in tag else tag
                        if role in self.color_map:
                            best_role = role
                            best_color = self.color_map[role]
            
            # Create highlighted span
            highlighted_words.append(
                f'<span style="background-color: {best_color}; padding: 2px 4px; border-radius: 3px; margin: 1px;">{word}</span>'
            )
        
        return ' '.join(highlighted_words)
    
    def create_performance_metrics(self, srl_results: List[Dict[str, Any]]) -> go.Figure:
        """Create performance metrics visualization"""
        if not srl_results:
            return go.Figure()
        
        # Calculate metrics
        processing_times = [r.get('processing_time', 0) for r in srl_results if 'processing_time' in r]
        verb_counts = [len(r.get('verbs', [])) for r in srl_results]
        
        metrics = {
            'Avg Processing Time': np.mean(processing_times) if processing_times else 0,
            'Total Verbs Found': sum(verb_counts),
            'Avg Verbs per Sentence': np.mean(verb_counts) if verb_counts else 0,
            'Total Analyses': len(srl_results)
        }
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(metrics.keys()),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Processing time gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['Avg Processing Time'],
            title={'text': "Avg Processing Time (s)"},
            gauge={'axis': {'range': [None, max(processing_times) if processing_times else 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 1], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 1}}),
            row=1, col=1
        )
        
        # Verb count gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['Total Verbs Found'],
            title={'text': "Total Verbs Found"},
            gauge={'axis': {'range': [None, max(verb_counts) * len(srl_results) if verb_counts else 1]},
                   'bar': {'color': "darkgreen"},
                   'steps': [{'range': [0, 10], 'color': "lightgray"},
                            {'range': [10, 20], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 20}}),
            row=1, col=2
        )
        
        # Average verbs gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['Avg Verbs per Sentence'],
            title={'text': "Avg Verbs per Sentence"},
            gauge={'axis': {'range': [None, max(verb_counts) if verb_counts else 1]},
                   'bar': {'color': "darkorange"},
                   'steps': [{'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 2}}),
            row=2, col=1
        )
        
        # Total analyses gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics['Total Analyses'],
            title={'text': "Total Analyses"},
            gauge={'axis': {'range': [None, len(srl_results) * 2]},
                   'bar': {'color': "darkred"},
                   'steps': [{'range': [0, len(srl_results)], 'color': "lightgray"},
                            {'range': [len(srl_results), len(srl_results) * 2], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': len(srl_results)}}),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title="SRL Performance Metrics")
        
        return fig

def demo_visualizations():
    """Demo function to showcase visualization capabilities"""
    print("🎨 SRL Visualization Demo")
    print("=" * 40)
    
    # Sample SRL result
    sample_result = {
        'verbs': [
            {
                'verb': 'gave',
                'tags': ['B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARG2', 'B-ARGM-TMP', 'I-ARGM-TMP']
            }
        ],
        'words': ['John', 'gave', 'a', 'book', 'to', 'Mary', 'on', 'her', 'birthday']
    }
    
    visualizer = SRLVisualizer()
    
    # Test highlighting
    sentence = "John gave a book to Mary on her birthday."
    highlighted = visualizer.highlight_text_with_roles(sentence, sample_result)
    print(f"Highlighted text: {highlighted}")
    
    print("\nVisualization tools ready!")
    print("Available methods:")
    print("- create_dependency_tree()")
    print("- create_role_distribution_chart()")
    print("- create_role_timeline()")
    print("- create_model_comparison()")
    print("- create_role_heatmap()")
    print("- create_performance_metrics()")

if __name__ == "__main__":
    demo_visualizations()
