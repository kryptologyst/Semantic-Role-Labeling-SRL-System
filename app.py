"""
Streamlit Web UI for Semantic Role Labeling
Modern, interactive interface for SRL analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from typing import Dict, List, Any

# Import our modules
from modern_srl import SemanticRoleLabeler
from mock_database import MockDatabase

# Page configuration
st.set_page_config(
    page_title="Semantic Role Labeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .srl-result {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .verb-highlight {
        background-color: #ffeb3b;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .arg-highlight {
        background-color: #e3f2fd;
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_srl_system():
    """Load SRL system with caching"""
    return SemanticRoleLabeler()

@st.cache_data
def load_database():
    """Load database with caching"""
    return MockDatabase()

def create_role_visualization(srl_result: Dict[str, Any]) -> go.Figure:
    """Create interactive visualization of SRL results"""
    if not srl_result.get('verbs'):
        return go.Figure()
    
    fig = go.Figure()
    
    words = srl_result.get('words', [])
    verbs = srl_result.get('verbs', [])
    
    # Create a simple bar chart showing role distribution
    role_counts = {}
    for verb in verbs:
        tags = verb.get('tags', [])
        for tag in tags:
            if tag != 'O':
                role = tag.split('-')[-1] if '-' in tag else tag
                role_counts[role] = role_counts.get(role, 0) + 1
    
    if role_counts:
        fig.add_trace(go.Bar(
            x=list(role_counts.keys()),
            y=list(role_counts.values()),
            marker_color='lightblue',
            text=list(role_counts.values()),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Semantic Role Distribution",
            xaxis_title="Role Type",
            yaxis_title="Count",
            height=400
        )
    
    return fig

def highlight_srl_in_text(sentence: str, srl_result: Dict[str, Any]) -> str:
    """Create HTML with highlighted SRL roles"""
    words = sentence.split()
    verbs = srl_result.get('verbs', [])
    
    if not verbs:
        return sentence
    
    # Simple highlighting (in a real implementation, you'd parse the tags properly)
    highlighted_words = []
    for word in words:
        # Check if word is a verb
        is_verb = any(word.lower() in verb.get('verb', '').lower() for verb in verbs)
        
        if is_verb:
            highlighted_words.append(f'<span class="verb-highlight">{word}</span>')
        else:
            highlighted_words.append(f'<span class="arg-highlight">{word}</span>')
    
    return ' '.join(highlighted_words)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Semantic Role Labeling</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze semantic roles in text using state-of-the-art NLP models**")
    
    # Load systems
    with st.spinner("Loading SRL system..."):
        srl = load_srl_system()
        db = load_database()
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Model selection
    model_info = srl.get_model_info()
    available_models = list(model_info.keys())
    
    if available_models:
        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            help="Choose the SRL model to use for analysis"
        )
        
        st.sidebar.markdown("### Model Information")
        if selected_model in model_info:
            info = model_info[selected_model]
            st.sidebar.write(f"**Type:** {info['type']}")
            st.sidebar.write(f"**Description:** {info['description']}")
            st.sidebar.write(f"**Status:** {info['status']}")
    else:
        st.sidebar.error("No SRL models available!")
        st.stop()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Text", "📚 Sample Sentences", "📊 Statistics", "🔄 Batch Processing"])
    
    with tab1:
        st.header("Single Sentence Analysis")
        
        # Text input
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sentence = st.text_area(
                "Enter a sentence to analyze:",
                value="John gave a book to Mary on her birthday.",
                height=100,
                help="Enter any English sentence for semantic role labeling"
            )
        
        with col2:
            st.markdown("### Quick Examples")
            examples = [
                "The chef prepared a delicious meal.",
                "Sarah bought flowers yesterday.",
                "The teacher explained the concept clearly."
            ]
            
            for example in examples:
                if st.button(f"📝 {example[:30]}...", key=f"example_{example}"):
                    st.session_state.example_sentence = example
            
            if 'example_sentence' in st.session_state:
                sentence = st.session_state.example_sentence
        
        # Analysis button
        if st.button("🚀 Analyze Sentence", type="primary"):
            if sentence.strip():
                with st.spinner("Analyzing sentence..."):
                    try:
                        result = srl.analyze_sentence(sentence, selected_model)
                        
                        # Display results
                        st.success(f"Analysis completed in {result['processing_time']:.3f} seconds!")
                        
                        # Highlighted text
                        st.markdown("### 📝 Highlighted Text")
                        highlighted = highlight_srl_in_text(sentence, result)
                        st.markdown(highlighted, unsafe_allow_html=True)
                        
                        # Detailed results
                        st.markdown("### 🔍 Detailed Analysis")
                        
                        if result.get('verbs'):
                            for i, verb in enumerate(result['verbs']):
                                with st.expander(f"Verb {i+1}: {verb.get('verb', 'N/A')}"):
                                    st.write(f"**Verb:** {verb.get('verb', 'N/A')}")
                                    if 'tags' in verb:
                                        st.write(f"**Tags:** {' '.join(verb['tags'])}")
                                    if 'description' in verb:
                                        st.write(f"**Description:** {verb['description']}")
                        
                        # Visualization
                        st.markdown("### 📊 Role Distribution")
                        fig = create_role_visualization(result)
                        if fig.data:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Save to database
                        srl.save_result_to_db(sentence, result)
                        st.info("Result saved to database!")
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
            else:
                st.warning("Please enter a sentence to analyze.")
    
    with tab2:
        st.header("Sample Sentences from Database")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category = st.selectbox(
                "Category",
                ["All", "simple", "complex", "modal", "political", "creative", "medical", "technical", "legal", "agricultural", "aviation", "scientific", "sports"],
                help="Filter sentences by category"
            )
        
        with col2:
            difficulty = st.selectbox(
                "Difficulty",
                ["All", "easy", "medium", "hard"],
                help="Filter sentences by difficulty level"
            )
        
        with col3:
            limit = st.slider("Number of sentences", 1, 20, 5)
        
        # Get sentences
        category_filter = None if category == "All" else category
        difficulty_filter = None if difficulty == "All" else difficulty
        
        sentences = db.get_sentences(category=category_filter, difficulty=difficulty_filter, limit=limit)
        
        if sentences:
            st.markdown(f"### Found {len(sentences)} sentences")
            
            for i, sentence in enumerate(sentences):
                with st.expander(f"Sentence {i+1}: {sentence['text'][:50]}..."):
                    st.write(f"**Text:** {sentence['text']}")
                    st.write(f"**Category:** {sentence['category']}")
                    st.write(f"**Difficulty:** {sentence['difficulty']}")
                    
                    if st.button(f"Analyze", key=f"analyze_{sentence['id']}"):
                        with st.spinner("Analyzing..."):
                            try:
                                result = srl.analyze_sentence(sentence['text'], selected_model)
                                
                                st.success("Analysis completed!")
                                
                                if result.get('verbs'):
                                    for verb in result['verbs']:
                                        st.write(f"**Verb:** {verb.get('verb', 'N/A')}")
                                        if 'tags' in verb:
                                            st.write(f"**Tags:** {' '.join(verb['tags'])}")
                                
                                srl.save_result_to_db(sentence['text'], result)
                                
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")
        else:
            st.warning("No sentences found with the selected filters.")
    
    with tab3:
        st.header("Database Statistics")
        
        stats = db.get_statistics()
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sentences", stats['total_sentences'])
        
        with col2:
            st.metric("Total Analyses", stats['total_srl_results'])
        
        with col3:
            st.metric("Categories", len(stats['categories']))
        
        with col4:
            st.metric("Models Used", len(stats['models']))
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sentences by Category")
            if stats['categories']:
                fig_cat = px.pie(
                    values=list(stats['categories'].values()),
                    names=list(stats['categories'].keys()),
                    title="Sentence Distribution by Category"
                )
                st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            st.markdown("### Sentences by Difficulty")
            if stats['difficulties']:
                fig_diff = px.bar(
                    x=list(stats['difficulties'].keys()),
                    y=list(stats['difficulties'].values()),
                    title="Sentence Distribution by Difficulty",
                    color=list(stats['difficulties'].values()),
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_diff, use_container_width=True)
        
        # Recent analyses
        st.markdown("### Recent Analyses")
        recent_results = db.get_srl_results(limit=10)
        
        if recent_results:
            df = pd.DataFrame(recent_results)
            st.dataframe(
                df[['sentence_text', 'model_name', 'processing_time', 'created_at']],
                use_container_width=True
            )
        else:
            st.info("No analyses found in database.")
    
    with tab4:
        st.header("Batch Processing")
        
        st.markdown("Upload a text file or enter multiple sentences for batch analysis.")
        
        # File upload
        uploaded_file = st.file_uploader("Upload text file", type=['txt'])
        
        if uploaded_file:
            content = uploaded_file.read().decode('utf-8')
            sentences = [line.strip() for line in content.split('\n') if line.strip()]
        else:
            # Manual input
            batch_text = st.text_area(
                "Enter multiple sentences (one per line):",
                height=200,
                help="Enter sentences separated by new lines"
            )
            sentences = [line.strip() for line in batch_text.split('\n') if line.strip()]
        
        if sentences:
            st.markdown(f"### Found {len(sentences)} sentences")
            
            if st.button("🚀 Analyze All Sentences", type="primary"):
                progress_bar = st.progress(0)
                results = []
                
                for i, sentence in enumerate(sentences):
                    try:
                        result = srl.analyze_sentence(sentence, selected_model)
                        results.append(result)
                        srl.save_result_to_db(sentence, result)
                    except Exception as e:
                        results.append({'error': str(e), 'sentence': sentence})
                    
                    progress_bar.progress((i + 1) / len(sentences))
                
                st.success(f"Batch analysis completed! Processed {len(sentences)} sentences.")
                
                # Display results summary
                successful = len([r for r in results if 'error' not in r])
                failed = len([r for r in results if 'error' in r])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Successful", successful)
                with col2:
                    st.metric("Failed", failed)
                
                # Show detailed results
                with st.expander("View Detailed Results"):
                    for i, result in enumerate(results):
                        if 'error' in result:
                            st.error(f"Sentence {i+1}: {result['error']}")
                        else:
                            st.success(f"Sentence {i+1}: {result['sentence']}")
                            if result.get('verbs'):
                                for verb in result['verbs']:
                                    st.write(f"  • Verb: {verb.get('verb', 'N/A')}")

if __name__ == "__main__":
    main()
