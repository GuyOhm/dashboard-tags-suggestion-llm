# 🏷️ StackOverflow Tags Analysis & Prediction Dashboard

Interactive dashboard for exploratory analysis of StackOverflow textual data and tag prediction using Large Language Models (LLM).

## 📋 Features

### 📊 Exploratory Analysis
- **Text length analysis**: Distribution of words and characters in titles and bodies
- **Word frequency**: Identification of most frequent terms with interactive visualizations
- **WordCloud**: Word clouds for titles and bodies of questions
- **Tags analysis**: Tag popularity and statistics
- **Complete dataset analysis**: Automatic analysis on the entire dataset (train + test, ~50K questions)

### 🤖 Tag Prediction
- **OpenAI GPT Integration**: Using GPT-4o-mini with optimized configuration
- **Few-shot prompting**: Using the 3 most popular questions as examples
- **Interactive interface**: Real-time testing with predefined questions or manual input
- **Secure configuration**: API key from environment variables only

## 🚀 Installation

### Prerequisites
- Python 3.11+
- OpenAI API key (for tag prediction)

### Environment Setup

```bash
# Navigate to project directory
cd dashboard-tags-suggestion-llm

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Create environment file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

### Data Preparation

Data files must be placed in the `data/` folder:
- `data/train.csv`: Training data
- `data/test.csv`: Test data

Expected column format:
- `Id`: Unique question identifier
- `Title`: Question title
- `Body`: Question body
- `Tags`: Associated tags
- `Score`, `ViewCount`, `AnswerCount`: Engagement metrics

## 🖥️ Usage

### Running the Dashboard

```bash
# With uv
uv run streamlit run main.py

# Or with shell script
./run_dashboard.sh

# Or directly
streamlit run main.py
```

The dashboard will be accessible at: `http://localhost:8501`

### Navigation

1. **📊 Exploratory Analysis**: 
   - View comprehensive analysis of the complete dataset (50K questions)
   - Explore different analysis tabs
   - Consult statistics and interactive charts

2. **🤖 Tag Prediction**:
   - Select from 10 predefined test questions or manual input
   - Get automatic tag predictions using GPT-4o-mini
   - View few-shot examples used for prediction

## ♿ Accessibility (WCAG)

The dashboard integrates several accessibility features:

- **High contrast**: Texts and graphics respect WCAG AA contrast ratios
- **Keyboard navigation**: All elements are keyboard accessible
- **Alternative texts**: Descriptions available for graphics
- **Semantic structure**: Appropriate use of HTML tags
- **Screen reader support**: Compatible with assistive technologies
- **Visible focus**: Clear visual indicators for navigation

## 🏗️ Architecture

```
dashboard-tags-suggestion-llm/
├── src/
│   └── lib/
│       ├── data_loader.py       # Data loading and preprocessing
│       ├── exploratory_analysis.py # Visualization generation
│       └── LLMClassifier.py     # LLM interface
├── data/                        # CSV data files
├── analysis/                    # Saved analysis results
├── main.py                     # Main Streamlit application
├── pyproject.toml             # Dependencies configuration
├── run_dashboard.sh           # Dashboard launcher script
└── config_git_lfs.sh         # Git LFS setup script
```

### Main Modules

#### `DataLoader`
- Optimized data loading with automatic full dataset analysis
- Statistical text analysis (length, word frequency)
- Tag analysis and statistics
- Persistent analysis results (JSON, Pickle, TXT)

#### `ExploratoryAnalysis`
- Interactive Plotly graphics generation
- WordCloud creation
- Multi-dimensional visualizations
- Streamlined analysis focused on key insights

#### `LLMClassifier`
- Unified interface for LLM providers (OpenAI)
- Few-shot prompting strategy
- Robust JSON response parsing

## 📊 Available Analysis Types

### Text Statistical Analysis
1. **Length distribution**: Histograms of words/characters in titles and bodies
2. **Text insights**: Average metrics for the complete dataset

### Frequency Analysis
1. **Top frequent words**: Interactive horizontal bar charts
2. **WordClouds**: Visualization of dominant terms in titles and bodies

### Tag Analysis
1. **Popularity**: Ranking of most used tags
2. **Statistics**: Global metrics on tag usage

## 🔧 Advanced Configuration

### Environment Variables

```env
# OpenAI API
OPENAI_API_KEY=sk-...
```

### Features

- **Automatic analysis**: Complete dataset analysis on first run
- **LLM Model**: Fixed to GPT-4o-mini for optimal performance
- **Prompting strategy**: Few-shot with 3 most popular questions as examples
- **Secure design**: API key from environment only, no UI configuration
- **Git LFS support**: For managing large data files

## 📈 Performance

- **Data loading**: Optimized for complete dataset analysis (~50K questions)
- **Caching**: Streamlit session state to avoid recalculations
- **Persistent results**: Analysis results saved in multiple formats (JSON, Pickle, TXT)
- **Visualizations**: Optimized Plotly rendering for interactivity
- **API calls**: Error handling and robust response parsing

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is developed as part of Project 7 - Machine Learning Engineer.

## 🆘 Support

For any questions or issues:
- Check the APIs documentation
- Verify API key configuration
- Ensure data is in the correct format
- Review the analysis folder for saved results

## 📚 Technologies Used

- **Frontend**: Streamlit
- **Visualizations**: Plotly, Matplotlib, WordCloud
- **Data Science**: Pandas, NumPy, scikit-learn
- **NLP**: NLTK
- **LLM**: OpenAI API
- **Styling**: Custom CSS for accessibility
- **Package Management**: uv
- **Version Control**: Git with LFS support
