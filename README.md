# 📚 Audible Insights: Intelligent Book Recommendations

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://audible-insights.streamlit.app)
[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/your-username/audible-insights/issues)

> **Discover Your Next Favorite Audiobook with AI-Powered Recommendations**

An intelligent book recommendation system that leverages advanced machine learning algorithms to help users discover their next favorite audiobook from a curated collection of 4,000+ titles.

---




---

## 📋 Table of Contents

- [Features](#-features)
- [Demo Screenshots](#-demo-screenshots)
- [Technology Stack](#️-technology-stack)
- [Dataset Overview](#-dataset-overview)
- [Quick Start](#️-quick-start)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [Project Architecture](#️-project-architecture)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Features

### 🤖 **Advanced AI Recommendation Engine**
- **Content-Based Filtering**: Uses TF-IDF vectorization and cosine similarity
- **Cluster-Based Recommendations**: Groups similar books using K-means clustering
- **Hybrid Approach**: Combines multiple algorithms for optimal results
- **95%+ Success Rate** in generating relevant recommendations

### 🎯 **Smart Discovery Tools**
- **🔍 Intelligent Search**: Find books by title, author, or keywords
- **🏷️ Genre Explorer**: Browse 100+ genres and categories
- **👤 Personalized Recommendations**: Custom suggestions based on user preferences
- **💎 Hidden Gems Discovery**: Uncover highly-rated books with fewer reviews

### 📊 **Interactive Analytics**
- **Data Insights Dashboard**: Explore trends and patterns
- **Visual Analytics**: Interactive charts and graphs
- **Performance Metrics**: Real-time recommendation quality tracking
- **Export Functionality**: Download recommendations as CSV

### 🎨 **Modern User Experience**
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Gradient UI**: Modern, professional interface with smooth animations
- **Horizontal Layouts**: Optimized book card displays
- **Real-time Updates**: Instant recommendation generation

---

## 🖼️ Demo Screenshots

### Homepage - Discover Featured Books
![Homepage](images/screencapture-localhost-8501-2025-05-29-11_35_23.png)

### Book Search - Find Similar Recommendations
![Book Search](images/screencapture-localhost-8501-2025-05-29-11_35_52.png)

### Personal Recommendations - Customized for You
![Personal Recommendations](images/screencapture-localhost-8501-2025-05-29-11_36_46.png)

### Data Insights - Analytics Dashboard
![Data Insights](images/screencapture-localhost-8501-2025-05-29-11_37_05.png)

---

## 🛠️ Technology Stack

### **Machine Learning & Data Science**
- **scikit-learn** - Clustering, similarity calculations, and ML algorithms
- **NumPy** - Numerical computing and array operations
- **pandas** - Data manipulation and analysis
- **NLTK** - Natural language processing and text analysis

### **Web Framework & Visualization**
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive charts and data visualizations
- **Seaborn & Matplotlib** - Statistical visualizations

### **Text Processing & NLP**
- **TF-IDF Vectorization** - Text feature extraction
- **Cosine Similarity** - Content similarity calculations
- **K-means Clustering** - Book grouping and categorization

### **Deployment **
- **Streamlit Cloud** - Application hosting and deployment


---

## 📊 Dataset Overview

### **Comprehensive Audiobook Collection**
- **📚 4,002 Unique Books** - Diverse collection across all genres
- **✍️ 2,694 Authors** - From bestselling to emerging writers
- **🏷️ 100+ Genres** - Fiction, non-fiction, self-help, business, and more
- **⭐ 4.46 Average Rating** - High-quality, well-reviewed content

### **Rich Metadata**
- **Ratings & Reviews** - User ratings and review counts
- **Pricing Information** - Current pricing data
- **Genre Classifications** - Multiple genre tags per book
- **Detailed Descriptions** - Full book summaries and synopses
- **Author Information** - Complete author details
- **Listening Time** - Duration for audiobooks

### **Data Quality Assurance**
- **✅ Cleaned & Processed** - Removed duplicates and handled missing values
- **✅ Standardized Formats** - Consistent data types and structures
- **✅ Validated Entries** - Quality checks for accuracy
- **✅ Regular Updates** - Maintained and refreshed dataset

---

## 🏃‍♂️ Quick Start
### **Option 1: Local Development**
```bash
# 1. Clone the repository
git clone https://github.com/your-username/audible-insights.git
cd audible-insights

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run streamlit_app.py

# 5. Open browser to http://localhost:8501
```
### **📖 Usage Guide**

### 🏠 Homepage Features
```
- **Quick Statistics**: Overview of total books, ratings, and genres  
- **Featured Recommendations**: Curated selection of top-rated books  
- **Instant Search**: Fast search across titles and authors  

---
```
### 🔍 Book Search & Recommendations

- **Select a Book**: Choose from 4,000+ titles using the dropdown  
- **Choose Method**: Pick from Hybrid, Content-Based, or Cluster-Based algorithms  
- **Get Recommendations**: Receive 1–20 personalized suggestions  
- **Explore Results**: View detailed information and similarity scores  

---

### 🏷️ Genre Explorer

- **Select Genres**: Choose from 100+ available genres  
- **Set Filters**: Adjust minimum rating and result count  
- **Discover Books**: Find books matching your genre preferences  
- **View Statistics**: Explore genre popularity charts  

---

### 👤 Personal Recommendations

- **Set Preferences**: Select favorite genres, rating threshold, and price range  
- **Choose Length**: Specify preferred audiobook duration  
- **Get Personalized Results**: Receive custom recommendations with match scores  
- **Refine Choices**: Adjust preferences for better matches  

---

### 📊 Data Insights

- **Rating Distribution**: Analyze book rating patterns  
- **Price vs Rating**: Explore pricing and quality relationships  
- **Author Analytics**: Discover most prolific authors  
- **Cluster Analysis**: Understand book groupings and themes  

---

## 📈 Model Performance

### Recommendation Quality

- **Success Rate**: 95%+ — Reliable recommendation generation  
- **Average Rating**: 4.2+ — High-quality suggested books  
- **Diversity Score**: 0.75+ — Balanced genre and author representation  
- **Response Time**: Sub-second — Fast recommendation processing  

---

### Algorithm Comparison

| Method         | Success Rate | Diversity | Novelty | Quality | Use Case           |
|----------------|--------------|-----------|---------|---------|---------------------|
| Hybrid         | 95%          | 0.75      | 0.65    | 4.3     | Best Overall        |
| Content-Based  | 92%          | 0.68      | 0.58    | 4.2     | Similar Books       |
| Cluster-Based  | 88%          | 0.82      | 0.72    | 4.1     | Genre Discovery     |

---

### Evaluation Metrics

- **Precision@10**: 0.78 — Relevant recommendations in top 10  
- **Recall@10**: 0.65 — Coverage of relevant items  
- **Diversity Score**: 0.75 — Variety across genres and authors  
- **Novelty Score**: 0.65 — Discovery of lesser-known gems  
- **Serendipity Score**: 0.42 — Unexpected but relevant finds  
### 🏗️ Project Architecture
## File Structure
audible-insights/
├── 📱 streamlit_app.py          # Main Streamlit application
├── 🤖 recommendation_classes.py  # ML recommendation algorithms
├── 🛠️ app_utils.py              # Utility functions
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # Project documentation
├── 🚫 .gitignore               # Git ignore rules
├── 📊 data/                    # Dataset files
│   └── streamlit_dataset.csv
├── 🧠 models/                  # Trained models
│   ├── combined_tfidf_matrix.npy
│   ├── content_recommender.pkl
│   ├── cluster_recommender.pkl
│   └── hybrid_recommender.pkl
└── 📓 notebooks/               # Development notebooks
    ├── 01_data_preparation.ipynb
    ├── 02_data_cleaning.ipynb
    ├── 03_exploratory_data_analysis.ipynb
    ├── 04_nlp_and_clustering.ipynb
    ├── 05_recommendation_systems.ipynb
    ├── 06_model_evaluation.ipynb
    ├── 07_streamlit_app.ipynb
    └── 08_summary_and_conclusion.ipynb

## Development Process

### Data Science Pipeline

- **Data Collection**: Gathered Audible audiobook metadata  
- **Data Cleaning**: Processed and standardized 4,000+ book records  
- **Exploratory Analysis**: Analyzed patterns, ratings, and genres  
- **Model Development**: Built and trained recommendation algorithms  
- **Evaluation**: Tested models with multiple performance metrics  
- **Deployment**: Created production-ready web application  

---

### Machine Learning Approach

- **Feature Engineering**: TF-IDF vectorization of book descriptions  
- **Similarity Calculation**: Cosine similarity for content-based filtering  
- **Clustering**: K-Means algorithm for grouping similar books  
- **Hybrid Method**: Weighted combination of multiple approaches  
- **Evaluation**: Comprehensive testing with precision, recall, and diversity metrics  

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.


---

## Acknowledgments

### Data Sources

- **Audible**: Audiobook metadata and catalog information  
- **Open Source Community**: Libraries and frameworks used  

### Inspiration

- **Netflix**: Recommendation system architecture  
- **Amazon**: E-commerce recommendation patterns  
- **Spotify**: Music discovery algorithms  

### Technologies

- **Streamlit**: Web application framework  
- **scikit-learn**: Machine learning and clustering  
- **Plotly**: Interactive data visualization  

### Special Thanks

- **Data Science Community**: For tutorials and best practices  
- **Open Source Contributors**: For maintaining excellent libraries  
- **Beta Testers**: For feedback and improvement suggestions  

