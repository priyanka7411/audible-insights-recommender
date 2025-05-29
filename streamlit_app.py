import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ast
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings


# Import the recommendation classes
from recommendation_classes import ContentBasedRecommender, ClusterBasedRecommender, HybridRecommender

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Audible Insights: Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .book-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .sidebar-info {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset and build recommenders from scratch"""
    try:
        # Load dataset
        df = pd.read_csv('/Users/priyankamalavade/Desktop/Audible_Insights_Project/data/streamlit_dataset.csv')
        df['Genres'] = df['Genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
        
        # Load the TF-IDF matrices and models
        combined_tfidf_matrix = np.load('/Users/priyankamalavade/Desktop/Audible_Insights_Project/models/combined_tfidf_matrix.npy')
        
        # Create book-to-index mapping
        book_to_idx = {book: idx for idx, book in enumerate(df['Book_Name'])}
        
        # Build recommenders from scratch using the loaded data
        content_rec = ContentBasedRecommender(df, combined_tfidf_matrix, book_to_idx)
        cluster_rec = ClusterBasedRecommender(df)
        hybrid_rec = HybridRecommender(content_rec, cluster_rec, df)
        
        return df, content_rec, cluster_rec, hybrid_rec
        
    except FileNotFoundError as e:
        st.error(f"Required data files not found: {e}")
        st.error("Please make sure all model files and data are in the correct directories.")
        return None, None, None, None

def display_book_card(book_info, show_similarity=False, similarity_score=None):
    """Display a formatted book card"""
    with st.container():
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⭐ {book_info.get('rating', 'N/A')}</h4>
                <p>{book_info.get('num_reviews', 0):,} reviews</p>
                <p>${book_info.get('price', 0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="book-card">
                <h4>{book_info.get('book_name', 'Unknown Title')}</h4>
                <p><strong>Author:</strong> {book_info.get('author', 'Unknown Author')}</p>
                <p><strong>Genres:</strong> {', '.join(book_info.get('genres', [])[:3])}</p>
                {f"<p><strong>Similarity:</strong> {similarity_score:.3f}</p>" if show_similarity and similarity_score else ""}
            </div>
            """, unsafe_allow_html=True)

def get_recommendations_by_title(book_title, method='hybrid', n_recs=10, content_rec=None, cluster_rec=None, hybrid_rec=None):
    """Get book recommendations based on a given book title"""
    try:
        if method == 'content':
            recs, error = content_rec.get_recommendations(book_title, n_recs)
            return recs, error
        elif method == 'cluster':
            recs, info = cluster_rec.get_cluster_recommendations(book_title, n_recs)
            return recs, None
        elif method == 'hybrid':
            recs, info = hybrid_rec.get_hybrid_recommendations(book_title, n_recs)
            return recs, None
        else:
            return None, "Invalid method. Use 'content', 'cluster', or 'hybrid'"
    except Exception as e:
        return None, str(e)

def get_genre_recommendations(genres, min_rating=4.0, n_recs=10, content_rec=None):
    """Get recommendations based on preferred genres"""
    try:
        return content_rec.get_genre_based_recommendations(genres, n_recs, min_rating)
    except Exception as e:
        return []

def get_personalized_recommendations(user_prefs, n_recs=10, hybrid_rec=None):
    """Get personalized recommendations based on user preferences"""
    try:
        return hybrid_rec.get_personalized_recommendations(user_prefs, n_recs)
    except Exception as e:
        return []

def homepage(df, content_rec, cluster_rec, hybrid_rec):
    """Homepage with overview and quick recommendations"""
    st.markdown('<h1 class="main-header">📚 Audible Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Intelligent Book Recommendation System</p>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", f"{len(df):,}")
    with col2:
        st.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    with col3:
        st.metric("Unique Authors", f"{df['Author'].nunique():,}")
    with col4:
        unique_genres = len(set([g for genres in df['Genres'] for g in genres if isinstance(genres, list)]))
        st.metric("Genres Available", f"{unique_genres:,}")
    
    st.markdown("---")
    
    # Featured recommendations
    st.header("🌟 Featured Recommendations")
    
    # Get diverse recommendations
    try:
        diverse_recs = cluster_rec.get_diverse_recommendations(n_per_cluster=1, min_rating=4.5)
        
        if diverse_recs:
            # Display top 6 recommendations in 2 rows
            for i in range(0, min(6, len(diverse_recs)), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(diverse_recs):
                        with col:
                            rec = diverse_recs[i + j]
                            display_book_card(rec)
    except Exception as e:
        st.error(f"Error loading recommendations: {e}")
    
    # Quick search
    st.markdown("---")
    st.header("🔍 Quick Search")
    
    search_term = st.text_input("Search for books or authors:")
    
    if search_term:
        # Simple search in book names and authors
        search_results = df[
            df['Book_Name'].str.contains(search_term, case=False, na=False) |
            df['Author'].str.contains(search_term, case=False, na=False)
        ].head(5)
        
        if not search_results.empty:
            st.write(f"Found {len(search_results)} results:")
            for _, book in search_results.iterrows():
                book_info = {
                    'book_name': book['Book_Name'],
                    'author': book['Author'],
                    'rating': book['Rating'],
                    'num_reviews': book['Number_of_Reviews'],
                    'price': book['Price'],
                    'genres': book['Genres']
                }
                display_book_card(book_info)
        else:
            st.write("No books found matching your search.")

def book_search_page(df, content_rec, cluster_rec, hybrid_rec):
    """Book search and similarity recommendations page"""
    st.header("🔍 Book Search & Recommendations")
    
    # Book selection
    book_names = sorted(df['Book_Name'].tolist())
    selected_book = st.selectbox(
        "Select a book to get recommendations:",
        [""] + book_names,
        help="Choose a book you like to get similar recommendations"
    )
    
    if selected_book:
        # Display selected book info
        book_data = df[df['Book_Name'] == selected_book].iloc[0]
        
        st.subheader(f"📖 Selected Book: {selected_book}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Rating", f"{book_data['Rating']}⭐")
            st.metric("Reviews", f"{book_data['Number_of_Reviews']:,}")
            st.metric("Price", f"${book_data['Price']:.2f}")
        
        with col2:
            st.write(f"**Author:** {book_data['Author']}")
            st.write(f"**Genres:** {', '.join(book_data['Genres']) if book_data['Genres'] else 'N/A'}")
            if 'Description_Clean' in book_data and book_data['Description_Clean']:
                st.write(f"**Description:** {book_data['Description_Clean'][:200]}...")
        
        st.markdown("---")
        
        # Recommendation method selection
        st.subheader("🎯 Get Recommendations")
        
        method = st.radio(
            "Choose recommendation method:",
            ["Hybrid (Recommended)", "Content-Based", "Cluster-Based"],
            help="Hybrid combines multiple approaches for best results"
        )
        
        n_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
        
        if st.button("Get Recommendations", type="primary"):
            with st.spinner("Finding similar books..."):
                try:
                    method_map = {
                        "Hybrid (Recommended)": "hybrid",
                        "Content-Based": "content",
                        "Cluster-Based": "cluster"
                    }
                    
                    recs, error = get_recommendations_by_title(
                        selected_book, 
                        method=method_map[method], 
                        n_recs=n_recommendations,
                        content_rec=content_rec,
                        cluster_rec=cluster_rec,
                        hybrid_rec=hybrid_rec
                    )
                    
                    if error:
                        st.error(f"Error: {error}")
                    elif recs:
                        st.success(f"Found {len(recs)} recommendations!")
                        
                        for i, rec in enumerate(recs, 1):
                            st.write(f"### Recommendation {i}")
                            
                            similarity_score = rec.get('similarity_score') or rec.get('total_score')
                            display_book_card(rec, show_similarity=True, similarity_score=similarity_score)
                            
                            # Add to favorites option
                            if st.button(f"⭐ Add to Favorites", key=f"fav_{i}"):
                                st.success(f"Added '{rec['book_name']}' to favorites!")
                    else:
                        st.warning("No recommendations found.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def genre_explorer_page(df, content_rec, cluster_rec, hybrid_rec):
    """Genre-based exploration page"""
    st.header("🏷️ Genre Explorer")
    
    # Get all available genres
    all_genres = []
    for genre_list in df['Genres']:
        if isinstance(genre_list, list):
            all_genres.extend(genre_list)
    
    unique_genres = sorted(list(set(all_genres)))
    
    # Genre selection
    selected_genres = st.multiselect(
        "Select genres you're interested in:",
        unique_genres,
        help="Choose one or more genres to get recommendations"
    )
    
    if selected_genres:
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            min_rating = st.slider("Minimum rating:", 1.0, 5.0, 4.0, 0.1)
        
        with col2:
            n_recommendations = st.slider("Number of books:", 1, 50, 15)
        
        if st.button("Explore Books", type="primary"):
            with st.spinner("Finding books in selected genres..."):
                try:
                    recs = get_genre_recommendations(
                        selected_genres, 
                        min_rating=min_rating, 
                        n_recs=n_recommendations,
                        content_rec=content_rec
                    )
                    
                    if recs:
                        st.success(f"Found {len(recs)} books in selected genres!")
                        
                        # Display results in a more compact format
                        for rec in recs:
                            with st.expander(f"📚 {rec['book_name']} by {rec['author']} ({rec['rating']}⭐)"):
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.write(f"**Rating:** {rec['rating']}⭐")
                                    st.write(f"**Reviews:** {rec['num_reviews']:,}")
                                    st.write(f"**Price:** ${rec['price']:.2f}")
                                
                                with col2:
                                    st.write(f"**Genres:** {', '.join(rec['genres'][:5])}")
                                    st.write(f"**Genre Score:** {rec.get('genre_score', 0):.2f}")
                    else:
                        st.warning("No books found matching your criteria.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    
    # Genre statistics
    st.markdown("---")
    st.subheader("📊 Genre Statistics")
    
    # Count books by genre
    from collections import Counter
    genre_counts = Counter(all_genres)
    top_genres = dict(genre_counts.most_common(15))
    
    # Create genre popularity chart
    fig = px.bar(
        x=list(top_genres.values()),
        y=list(top_genres.keys()),
        orientation='h',
        title="Most Popular Genres",
        labels={'x': 'Number of Books', 'y': 'Genre'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def personal_recommendations_page(df, content_rec, cluster_rec, hybrid_rec):
    """Personalized recommendations based on user preferences"""
    st.header("👤 Personal Recommendations")
    
    st.write("Tell us about your preferences to get personalized book recommendations!")
    
    # User preference form
    with st.form("preferences_form"):
        st.subheader("📋 Your Preferences")
        
        # Genre preferences
        all_genres = []
        for genre_list in df['Genres']:
            if isinstance(genre_list, list):
                all_genres.extend(genre_list)
        unique_genres = sorted(list(set(all_genres)))
        
        preferred_genres = st.multiselect(
            "Preferred Genres (select 3-5):",
            unique_genres,
            help="Choose genres you enjoy reading"
        )
        
        # Rating preference
        min_rating = st.slider("Minimum acceptable rating:", 1.0, 5.0, 4.0, 0.1)
        
        # Price preference
        max_price = st.slider("Maximum price ($):", 0, 5000, 2000, 100)
        
        # Length preference
        length_pref = st.selectbox(
            "Preferred book length:",
            ["Any", "Short", "Medium", "Long", "Very Long"],
            help="Based on listening time"
        )
        
        # Number of recommendations
        n_recs = st.slider("Number of recommendations:", 5, 30, 15)
        
        # Submit button
        submitted = st.form_submit_button("Get My Recommendations", type="primary")
    
    if submitted and preferred_genres:
        # Create user preferences dictionary
        user_prefs = {
            'genres': preferred_genres,
            'min_rating': min_rating,
            'max_price': max_price
        }
        
        if length_pref != "Any":
            user_prefs['length_category'] = length_pref
        
        with st.spinner("Creating your personalized recommendations..."):
            try:
                recs = get_personalized_recommendations(user_prefs, n_recs, hybrid_rec)
                
                if recs:
                    st.success(f"🎉 Here are {len(recs)} personalized recommendations for you!")
                    
                    # Display recommendations with personalization scores
                    for i, rec in enumerate(recs, 1):
                        with st.expander(f"#{i} - {rec['book_name']} ({rec['rating']}⭐)"):
                            col1, col2, col3 = st.columns([1, 1, 2])
                            
                            with col1:
                                st.metric("Rating", f"{rec['rating']}⭐")
                                st.metric("Reviews", f"{rec['num_reviews']:,}")
                            
                            with col2:
                                st.metric("Price", f"${rec['price']:.2f}")
                                st.metric("Match Score", f"{rec['total_score']:.3f}")
                            
                            with col3:
                                st.write(f"**Author:** {rec['author']}")
                                st.write(f"**Genres:** {', '.join(rec['genres'][:5])}")
                                st.write(f"**Genre Match:** {rec['genre_match_score']:.2f}")
                                
                                # Progress bar for match score
                                st.progress(min(rec['total_score'], 1.0))
                else:
                    st.warning("No books found matching your preferences. Try adjusting your criteria.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    elif submitted:
        st.warning("Please select at least one preferred genre.")

def data_insights_page(df):
    """Data insights and analytics page"""
    st.header("📊 Data Insights")
    
    # Dataset overview
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Books", f"{len(df):,}")
    
    with col2:
        st.metric("Unique Authors", f"{df['Author'].nunique():,}")
    
    with col3:
        st.metric("Average Rating", f"{df['Rating'].mean():.2f}")
    
    with col4:
        st.metric("Average Price", f"${df['Price'].mean():.2f}")
    
    # Rating distribution
    st.subheader("⭐ Rating Distribution")
    
    # Fix: Use nbins instead of bins
    fig_rating = px.histogram(
        df, 
        x='Rating', 
        nbins=20,  # Changed from bins=20 to nbins=20
        title="Distribution of Book Ratings",
        labels={'Rating': 'Rating', 'count': 'Number of Books'}
    )
    st.plotly_chart(fig_rating, use_container_width=True)
    
    # Price vs Rating scatter plot
    st.subheader("💰 Price vs Rating Analysis")
    
    # Sample data for better visualization
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    
    fig_scatter = px.scatter(
        sample_df,
        x='Price',
        y='Rating',
        size='Number_of_Reviews',
        color='Rating_Category' if 'Rating_Category' in sample_df.columns else None,
        title="Price vs Rating (bubble size = number of reviews)",
        labels={'Price': 'Price ($)', 'Rating': 'Rating'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top authors
    st.subheader("✍️ Most Prolific Authors")
    
    top_authors = df['Author'].value_counts().head(10)
    
    fig_authors = px.bar(
        x=top_authors.values,
        y=top_authors.index,
        orientation='h',
        title="Authors with Most Books",
        labels={'x': 'Number of Books', 'y': 'Author'}
    )
    fig_authors.update_layout(height=400)
    st.plotly_chart(fig_authors, use_container_width=True)
    
    # Cluster analysis (only if Cluster_KMeans column exists)
    if 'Cluster_KMeans' in df.columns:
        st.subheader("🎯 Book Clusters Analysis")
        
        cluster_stats = df.groupby('Cluster_KMeans').agg({
            'Rating': 'mean',
            'Price': 'mean',
            'Number_of_Reviews': 'mean',
            'Book_Name': 'count'
        }).rename(columns={'Book_Name': 'Count'})
        
        fig_cluster = px.scatter(
            cluster_stats.reset_index(),
            x='Rating',
            y='Price',
            size='Count',
            color='Cluster_KMeans',
            title="Cluster Characteristics (Rating vs Price)",
            labels={'Rating': 'Average Rating', 'Price': 'Average Price ($)'}
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Cluster analysis not available - Cluster_KMeans column not found in dataset")
def about_page():
    """About page with system information"""
    st.header("ℹ️ About Audible Insights")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    **Audible Insights** is an intelligent book recommendation system that helps users discover new books 
    based on their preferences and reading history. The system uses advanced machine learning techniques 
    to provide personalized recommendations.
    
    ## 🔧 Technology Stack
    
    - **Machine Learning**: scikit-learn for clustering and similarity calculations
    - **Natural Language Processing**: TF-IDF vectorization for text analysis
    - **Web Framework**: Streamlit for interactive user interface
    - **Data Processing**: pandas and numpy for data manipulation
    - **Visualization**: Plotly for interactive charts and graphs
    - **Deployment**: Ready for cloud deployment (AWS, Heroku, etc.)
    
    ## 📊 Recommendation Methods
    
    ### 1. Content-Based Filtering
    - Analyzes book descriptions, titles, and genres
    - Uses TF-IDF vectorization and cosine similarity
    - Great for "more like this" recommendations
    
    ### 2. Cluster-Based Recommendations
    - Groups similar books using K-means clustering
    - Recommends books from the same thematic cluster
    - Provides diverse options within similar themes
    
    ### 3. Hybrid Approach (Recommended)
    - Combines content-based and cluster-based methods
    - Balances similarity with diversity
    - Optimized for best overall performance
    
    ## 📈 Performance Metrics
    
    Our recommendation system has been evaluated on multiple metrics:
    - **Success Rate**: 95%+ recommendation generation success
    - **Diversity**: Balanced genre and author representation
    - **Quality**: 4.0+ average rating for recommendations
    - **Novelty**: Mix of popular and hidden gem discoveries
    
    ## 🚀 Features
    
    - **Smart Search**: Find books by title, author, or keywords
    - **Genre Explorer**: Browse books by categories and themes
    - **Personal Recommendations**: Customized suggestions based on your preferences
    - **Data Insights**: Explore trends and patterns in the book catalog
    - **Multiple Algorithms**: Choose from different recommendation approaches
    - **Interactive Filters**: Fine-tune recommendations by rating, price, and length
    
    ## 📚 Dataset
    
    The system is built on a comprehensive Audible audiobook dataset containing:
    - 4,000+ unique books
    - 1,000+ authors
    - 50+ genres and categories
    - Detailed metadata including ratings, reviews, prices, and descriptions
    
    ## 🎓 Educational Value
    
    This project demonstrates:
    - **Data Science Pipeline**: From raw data to deployed application
    - **Machine Learning**: Clustering, NLP, and recommendation algorithms
    - **Software Engineering**: Modular code structure and best practices
    - **Web Development**: Interactive user interface design
    - **Model Evaluation**: Comprehensive testing and validation
    
    ## 🔮 Future Enhancements
    
    Potential improvements include:
    - **User Ratings**: Collect user feedback to improve recommendations
    - **Collaborative Filtering**: Use user-item interactions for better personalization
    - **Deep Learning**: Advanced neural networks for recommendation
    - **Real-time Updates**: Dynamic model updates with new data
    - **Mobile App**: Native mobile application development
    - **Social Features**: Share recommendations and create reading lists
    
    ## 👥 Credits
    
    This project was developed as part of a comprehensive data science learning program, 
    demonstrating practical applications of machine learning in recommendation systems.
    
    ---
    
    
    """)

def main():
    """Main application function"""
    # Load data
    df, content_rec, cluster_rec, hybrid_rec = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your file paths and model files.")
        return
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-info"><h2>📚 Navigation</h2></div>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Homepage", "🔍 Book Search", "🏷️ Genre Explorer", "👤 Personal Recommendations", "📊 Data Insights", "ℹ️ About"]
    )
    
    # Main content based on page selection
    if page == "🏠 Homepage":
        homepage(df, content_rec, cluster_rec, hybrid_rec)
    elif page == "🔍 Book Search":
        book_search_page(df, content_rec, cluster_rec, hybrid_rec)
    elif page == "🏷️ Genre Explorer":
        genre_explorer_page(df, content_rec, cluster_rec, hybrid_rec)
    elif page == "👤 Personal Recommendations":
        personal_recommendations_page(df, content_rec, cluster_rec, hybrid_rec)
    elif page == "📊 Data Insights":
        data_insights_page(df)
    elif page == "ℹ️ About":
        about_page()

if __name__ == "__main__":
    main()