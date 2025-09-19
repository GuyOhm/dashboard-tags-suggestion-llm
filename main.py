import streamlit as st
import pandas as pd
# import plotly.graph_objects as go
from dotenv import load_dotenv
import os

from src.lib.data_loader import DataLoader
from src.lib.exploratory_analysis import ExploratoryAnalysis
from src.lib.LLMClassifier import LLMClassifier

# Load environment variables
load_dotenv()

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = None
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None

def load_or_generate_analysis():
    """Load existing analysis or generate a new one on the complete dataset"""
    if not st.session_state.data_loaded:
        data_loader = DataLoader()
        st.session_state.data_loader = data_loader
        
        # Try to load existing analysis
        with st.spinner("Searching for existing analysis..."):
            existing_analysis = data_loader.load_latest_analysis()
        
        if existing_analysis:
            # Existing analysis found
            st.success(f"✅ Existing analysis loaded: {existing_analysis['timestamp']}")
            st.info(f"📊 Analysis based on {existing_analysis['sample_size']} questions")
            st.session_state.analysis_data = existing_analysis
        else:
            # No existing analysis, generate on complete dataset
            st.info("ℹ️ No existing analysis found. Generating new analysis on complete dataset...")
            
            try:
                with st.spinner("🔄 Generating complete analysis (this may take a few minutes for 50K questions)..."):
                    analysis_data = data_loader.generate_full_analysis()
                    st.success(f"✅ New analysis generated on {analysis_data['sample_size']} questions")
                    st.session_state.analysis_data = analysis_data
            except Exception as e:
                st.error(f"❌ Error during analysis generation: {e}")
                return None
        
        st.session_state.data_loaded = True
            
    return st.session_state.analysis_data

def render_accessibility_info():
    """Display accessibility information"""
    with st.expander("ℹ️ Accessibility Information"):
        st.markdown("""
        **Integrated accessibility features:**
        - High contrast for texts and graphics
        - Keyboard navigation available
        - Alternative texts for graphics
        - Font size adjustable via browser settings
        - Screen reader support
        - Interactive graphics with textual descriptions
        """)

def render_exploratory_analysis():
    """Display the exploratory analysis section"""
    st.header("📊 Exploratory Analysis of Textual Data")
    st.markdown("**Complete analysis on the entire StackOverflow dataset (train + test)**")
    
    render_accessibility_info()
    
    # Load or generate analysis
    analysis_data = load_or_generate_analysis()
    
    if analysis_data:
        analyzer = ExploratoryAnalysis()
        
        # General statistics
        st.subheader("📈 Complete Dataset Statistics")
        stats = analysis_data['basic_stats']
        
        # First row of metrics
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        with metrics_col1:
            st.metric("Questions analyzed", f"{stats['combined_shape'][0]:,}")
        with metrics_col2:
            st.metric("Unique tags", analysis_data['tags_analysis']['total_unique_tags'])
        with metrics_col3:
            if 'source_distribution' in stats:
                train_count = stats['source_distribution'].get('train', 0)
                st.metric("Train questions", f"{train_count:,}")
        with metrics_col4:
            if 'source_distribution' in stats:
                test_count = stats['source_distribution'].get('test', 0)
                st.metric("Test questions", f"{test_count:,}")
        
        # Text statistics
        if 'text_analysis' in analysis_data and not analysis_data['text_analysis'].empty:
            text_data = analysis_data['text_analysis']
            st.write("**Text averages across the entire dataset:**")
            text_metrics_cols = st.columns(4)
            with text_metrics_cols[0]:
                st.metric("Words/title", f"{text_data['title_words'].mean():.1f}")
            with text_metrics_cols[1]:
                st.metric("Words/body", f"{text_data['body_words'].mean():.1f}")
            with text_metrics_cols[2]:
                st.metric("Tags/question", f"{analysis_data['tags_analysis']['avg_tags_per_question']:.1f}")
            with text_metrics_cols[3]:
                if 'timestamp' in analysis_data:
                    st.metric("Analysis date", analysis_data['timestamp'][:8])
        
        # Tabs to organize analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📏 Text Length", 
            "📝 Word Frequency", 
            "☁️ WordCloud", 
            "🏷️ Tags Analysis"
        ])
        
        with tab1:
            st.subheader("Text length analysis")
            st.markdown(f"""
            This analysis examines the distribution of title and body length for questions across the entire dataset 
            ({stats['combined_shape'][0]:,} questions).
            """)
            
            # Generate graphs
            text_figures = analyzer.create_text_length_analysis(analysis_data['text_analysis'])
            
            # Display distribution graph
            st.plotly_chart(text_figures['length_distribution'], width='stretch')
            
            # Text insights
            with st.expander("💡 Text length insights"):
                text_data = analysis_data['text_analysis']
                st.write(f"""
                - **Average title:** {text_data['title_words'].mean():.1f} words
                - **Average body:** {text_data['body_words'].mean():.1f} words
                - **Average total:** {text_data['total_words'].mean():.1f} words per question
                - **Title characters (average):** {text_data['title_chars'].mean():.0f} characters
                - **Body characters (average):** {text_data['body_chars'].mean():.0f} characters
                """)
        
        with tab2:
            st.subheader("Word frequency analysis")
            st.markdown(f"""
            Analysis of the most frequent words in question titles and bodies across the entire dataset 
            ({stats['combined_shape'][0]:,} questions).
            """)
            
            freq_figures = analyzer.create_word_frequency_analysis(
                analysis_data['word_freq_title'],
                analysis_data['word_freq_body']
            )
            
            st.plotly_chart(freq_figures['word_frequency'], width='stretch')
            
            # Top words in a table
            with st.expander("📋 Top 10 words by section"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Titles:**")
                    top_title = list(analysis_data['word_freq_title'].items())[:10]
                    st.dataframe(pd.DataFrame(top_title, columns=['Word', 'Frequency']))
                
                with col2:
                    st.write("**Bodies:**")
                    top_body = list(analysis_data['word_freq_body'].items())[:10]
                    st.dataframe(pd.DataFrame(top_body, columns=['Word', 'Frequency']))
        
        with tab3:
            st.subheader("Word clouds (WordCloud)")
            st.markdown(f"""
            Visualization of the most frequent words as word clouds
            for question titles and bodies across the entire dataset ({stats['combined_shape'][0]:,} questions).
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**WordCloud - Titles**")
                wordcloud_title = analyzer.generate_wordcloud(
                    analysis_data['word_freq_title'], 
                    "Most frequent words - Titles"
                )
                st.image(wordcloud_title, width='stretch')
            
            with col2:
                st.write("**WordCloud - Bodies**")
                wordcloud_body = analyzer.generate_wordcloud(
                    analysis_data['word_freq_body'],
                    "Most frequent words - Bodies"
                )
                st.image(wordcloud_body, width='stretch')
        
        with tab4:
            st.subheader("Tags analysis")
            st.markdown(f"""
            Analysis of the popularity of the most used tags to categorize StackOverflow questions
            across the entire dataset ({stats['combined_shape'][0]:,} questions).
            """)
            
            tags_figures = analyzer.create_tags_analysis(analysis_data['tags_analysis'])
            
            st.plotly_chart(tags_figures['top_tags'], width='stretch')
            
            # Tag statistics
            with st.expander("📊 Tag statistics"):
                tags_stats = analysis_data['tags_analysis']
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Unique tags", tags_stats['total_unique_tags'])
                with col2:
                    st.metric("Total occurrences", tags_stats['total_tag_occurrences'])
                with col3:
                    st.metric("Average per question", f"{tags_stats['avg_tags_per_question']:.1f}")
        

def render_tag_prediction():
    """Display the tag prediction section"""
    
    # Create centered layout with limited width
    col_left, col_right = st.columns([2, 2])
    
    with col_left:
        st.header("🤖 Tag Prediction with LLM")
        st.markdown("""
        **Automatic configuration:**
        - Model: GPT-4o-mini
        - Prompt type: Few-shot (with 3 examples of popular questions)
        - API Key: retrieved from environment variables
        """)
        
        # Check for API key presence
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("❌ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            return
        
        # Input area for testing
        st.subheader("📝 Prediction test")
        
        # Load test question samples
        @st.cache_data
        def load_test_samples():
            test_df = pd.read_csv("data/test.csv")
            # Take 10 varied questions (not necessarily the first ones)
            sample_questions = test_df.sample(n=10, random_state=42).reset_index(drop=True)
            return sample_questions
        
        test_samples = load_test_samples()
        
        # Create options list for dropdown
        question_options = ["Manual input"] + [
            f"Q{i+1}: {title[:80]}..." if len(title) > 80 else f"Q{i+1}: {title}"
            for i, title in enumerate(test_samples['Title'])
        ]
        
        # Dropdown to select predefined question
        selected_option = st.selectbox(
            "🔽 Choose a predefined question or manual input:",
            question_options,
            help="Select a question from the test dataset or choose 'Manual input'"
        )
        
        # Initialize default values
        default_title = ""
        default_body = ""
        
        # If a predefined question is selected
        if selected_option != "Manual input":
            question_index = question_options.index(selected_option) - 1  # -1 because "Manual input" is at position 0
            selected_question = test_samples.iloc[question_index]
            default_title = selected_question['Title']
            default_body = selected_question['Body']
        
        # Input fields (with default values if a question is selected)
        test_title = st.text_input(
            "Question title:",
            value=default_title,
            placeholder="Ex: How to implement authentication in React?"
        )
        
        test_body = st.text_area(
            "Question body:",
            value=default_body,
            placeholder="Describe your question in detail...",
            height=150
        )
        
        # Display real tags if a predefined question is selected
        if selected_option != "Manual input":
            question_index = question_options.index(selected_option) - 1
            selected_question = test_samples.iloc[question_index]
            st.info(f"🏷️ Real tags for this question: `{selected_question['Tags']}`")
        
        if st.button("🔍 Predict tags", type="primary"):
            if test_title and test_body:
                try:
                    with st.spinner("Prediction in progress..."):
                        # Load training data for examples
                        @st.cache_data
                        def load_train_examples():
                            train_df = pd.read_csv("data/train.csv")
                            # Select the 3 questions with most answers
                            examples = train_df.sort_values("AnswerCount", ascending=False).head(3)
                            return examples
                        
                        train_examples = load_train_examples()
                        
                        # Create DataFrame for test
                        test_df = pd.DataFrame({
                            'Id': [999999],
                            'Title': [test_title],
                            'Body': [test_body]
                        })
                        
                        # Initialize classifier with fixed parameters
                        classifier = LLMClassifier(
                            provider="openai",
                            model_name="gpt-4o-mini",
                            api_key=api_key
                        )
                        
                        # Make prediction with few-shot and examples
                        response = classifier.predict(
                            questions=test_df,
                            prompt_type="few_shot",
                            examples=train_examples
                        )
                        
                        parsed_response = classifier._parse_json_response(response)
                        
                        if not parsed_response.empty:
                            st.success("✅ Prediction successful!")
                            
                            # Display results
                            predicted_tags = parsed_response.iloc[0]['TagsList']
                            
                            st.subheader("🏷️ Predicted tags:")
                            
                            # Display tags as badges
                            if predicted_tags:
                                tag_cols = st.columns(min(len(predicted_tags), 5))  # Limit to max 5 columns
                                for i, tag in enumerate(predicted_tags[:5]):  # Display max 5 tags
                                    with tag_cols[i]:
                                        st.markdown(f"<div style='font-size: 18px; font-weight: bold; background-color: #f0f2f6; padding: 8px; border-radius: 5px; text-align: center;'><code>{tag}</code></div>", unsafe_allow_html=True)
                            
                            # Display examples used
                            with st.expander("📚 Examples used for few-shot"):
                                for idx, row in train_examples.iterrows():
                                    st.write(f"**Question {idx+1}:**")
                                    st.write(f"- **Title:** {row['Title']}")
                                    st.write(f"- **Tags:** {row['Tags']}")
                                    st.write(f"- **Answers:** {row['AnswerCount']}")
                                    st.write("---")
                            
                            # Display complete response
                            with st.expander("📋 Complete model response"):
                                st.json(parsed_response.to_dict())
                                
                        else:
                            st.error("❌ Error during prediction")
                            st.text("Raw response:")
                            st.text(response)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.text("Error details for debugging:")
                    st.text(str(e))
            else:
                st.warning("⚠️ Please fill in both the title and body of the question.")

def main():
    """Main Streamlit application function"""
    # Page configuration
    st.set_page_config(
        page_title="StackOverflow Tags Dashboard",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for accessibility
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e1e5e9;
        border-radius: 5px;
        padding: 1rem;
    }
    
    /* Contrast improvement */
    .stSelectbox label, .stTextInput label, .stTextArea label {
        font-weight: bold;
        color: #262730;
    }
    
    /* Visible focus for accessibility */
    button:focus, input:focus, select:focus, textarea:focus {
        outline: 2px solid #0066cc;
        outline-offset: 2px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session variables
    init_session_state()
    
    # Header
    st.title("🏷️ StackOverflow Tags Analysis & Prediction Dashboard")
    st.markdown("""
    Interactive dashboard for exploratory analysis of textual data 
    and tag prediction using Large Language Models (LLM).
    """)
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🧭 Navigation")
        
        page = st.selectbox(
            "Choose a section:",
            ["📊 Exploratory Analysis", "🤖 Tag Prediction"],
            help="Navigate between different sections of the dashboard"
        )
        
        st.markdown("---")
        st.markdown("""
        ### ℹ️ About
        This dashboard presents a comprehensive exploratory 
        analysis of StackOverflow data and allows 
        testing tag prediction with LLMs.
        
        **Features:**
        - Statistical text analysis
        - Interactive visualizations
        - WordClouds
        - Prediction with OpenAI GPT
        """)
        
        st.markdown("---")
        st.markdown("**Built with:** Streamlit, Plotly, OpenAI")
    
    # Display pages
    if page == "📊 Exploratory Analysis":
        render_exploratory_analysis()
    elif page == "🤖 Tag Prediction":
        render_tag_prediction()

if __name__ == "__main__":
    main()
