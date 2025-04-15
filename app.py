
# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main { background-color: #f4f6f9; font-family: 'Segoe UI', sans-serif; }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        transition: 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #145a86;
    }
    .stRadio > div {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .book-card {
        background-color: #ffffff;
        border-left: 6px solid #1f77b4;
        border-radius: 10px;
        padding: 15px 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    .book-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .header {
        color: #1f2b3f;
    }
    h1, h2, h3, h4 {
        color: #1f2b3f;
    }
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data and Models ---
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_book_data_with_clusters.csv")
    cosine_sim = joblib.load("cosine_sim.pkl")
    kmeans = joblib.load("kmeans_model.pkl")
    svd_model = joblib.load("svd_recommender_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return df, cosine_sim, kmeans, svd_model, tfidf

df, cosine_sim, kmeans, svd_model, tfidf = load_data()

# --- Sidebar ---
with st.sidebar:
    st.title("Audible Insights")
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Home", "Explore Books", "Recommendations", "Contact"])

# --- Home Page ---
if page == "Home":
    st.title("Audible Insights 🎧")
    st.markdown("""
    ### Discover Your Next Favorite Audiobook
    Explore hand-picked recommendations tailored to your preferences.
    ---
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📚 Content-Based")
        st.write("Find similar books based on content features.")
    with col2:
        st.markdown("#### 🔍 Cluster-Based")
        st.write("Discover books grouped by shared traits.")
    with col3:
        st.markdown("#### 🤝 Collaborative")
        st.write("Get suggestions based on similar users.")

    st.markdown("---")
    st.image("https://img.freepik.com/premium-vector/audiobook-app-concept-illustration_701961-164.jpg", 
             use_container_width=True, caption="Find your next listening adventure")
    

## ------ EDA ---- ##
# --- Explore Books ---
elif page == "Explore Books":
    st.title("Explore the Audible Dataset")
    st.markdown("""
    ### Visualize and Interact with Audiobook Insights
    Analyze book genres, authors, ratings, and discover hidden patterns.
    ---
    """)

    tabs = st.tabs(["Dataset Overview", "Visual Analysis", "Insights & Trends"])

    # --- Tab 1: Dataset Overview ---
    with tabs[0]:
        st.markdown("#### 📄 Dataset Snapshot")
        st.dataframe(df.head())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📊 Descriptive Statistics")
            st.dataframe(df.describe())
        with col2:
            st.markdown("#### ❌ Missing Values")
            st.dataframe(df.isnull().sum())

    # --- Tab 2: Visualizations ---
    with tabs[1]:
        st.markdown("#### 📈 Interactive Visualizations")
        chart_type = st.selectbox("Choose a visualization type:", [
            "Rating Distribution", "Top Genres", "Top Authors", "Ratings vs Reviews", "Correlation Heatmap"],
            index=0
        )

        st.markdown("---")

        if chart_type == "Rating Distribution":
            fig = px.box(df, x="Genre", y="Rating", color="Genre")
            fig.update_layout(title="Rating Distribution by Genre")
            st.plotly_chart(fig, use_container_width=True, key="rating_distribution")

        elif chart_type == "Top Genres":
            genre_counts = df['Genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            fig = px.bar(genre_counts.head(10), x='Genre', y='Count', color='Genre',
                         title="Top Genres")
            st.plotly_chart(fig, use_container_width=True, key="top_genres")

        elif chart_type == "Top Authors":
            authors = df['Author'].value_counts().reset_index().head(10)
            authors.columns = ['Author', 'Books']
            fig = px.bar(authors, x='Author', y='Books', color='Author',
                         title="Top 10 Authors by Book Count")
            st.plotly_chart(fig, use_container_width=True, key="top_authors")

        elif chart_type == "Ratings vs Reviews":
            fig = px.scatter(df, x='Number of Reviews', y='Rating', color='Genre',
                             hover_data=['Book Name'], size='Rating',
                             title="Ratings vs Reviews")
            fig.update_layout(xaxis_type="log")
            st.plotly_chart(fig, use_container_width=True, key="ratings_vs_reviews")

        else:  # Correlation Heatmap
            numerical = df[['Rating', 'Number of Reviews', 'Price', 'Listening Time (minutes)', 'Rank']]
            fig = px.imshow(numerical.corr(), text_auto=True, color_continuous_scale='Tealrose',
                            title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

    # --- Tab 3: Insights ---
    with tabs[2]:
        st.markdown("#### ❓ Data-Driven Q&A")

        question = st.selectbox("Select a question to explore:", [
            "What are the most popular genres?",
            "Which authors have the highest-rated books?",
            "What is the average rating distribution?",
            "What are the hidden gems (highly rated but few reviews)?",
            "Cluster view of books (sample)",
            "Author popularity vs rating",
            "Recommend top Sci-Fi books",
            "Recommend for Thriller lovers"
        ])

        st.markdown("---")

        if question == "What are the most popular genres?":
            genre_counts = df['Genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            fig = px.bar(genre_counts.head(10), x='Genre', y='Count', color='Genre',
                         title="Top Genres")
            st.plotly_chart(fig, use_container_width=True, key="popular_genres")

        elif question == "Which authors have the highest-rated books?":
            top_authors = df.sort_values(by='Rating', ascending=False).drop_duplicates('Author').head(10)
            fig = px.bar(top_authors, x='Author', y='Rating', color='Rating', hover_data=['Book Name'],
                         title="Top Rated Authors")
            st.plotly_chart(fig, use_container_width=True, key="top_rated_authors")

        elif question == "What is the average rating distribution?":
            fig = px.histogram(df, x='Rating', nbins=20, color='Genre',
                               title="Distribution of Book Ratings")
            st.plotly_chart(fig, use_container_width=True, key="avg_rating_distribution")

        elif question == "What are the hidden gems (highly rated but few reviews)?":
            gems = df[(df['Rating'] >= 4.5) & (df['Number of Reviews'] < 100)]
            st.dataframe(gems[['Book Name', 'Author', 'Rating', 'Number of Reviews']]
                         .sort_values(by='Rating', ascending=False).head(10))

        elif question == "Cluster view of books (sample)":
            cluster_sample = df[['Book Name', 'cluster']].groupby('cluster').apply(lambda x: x.head(2)).reset_index(drop=True)
            st.dataframe(cluster_sample)

        elif question == "Author popularity vs rating":
            author_stats = df.groupby('Author').agg({'Rating': 'mean', 'Book Name': 'count'}).reset_index()
            author_stats.columns = ['Author', 'Avg Rating', 'Book Count']
            fig = px.scatter(author_stats, x='Book Count', y='Avg Rating', size='Avg Rating', color='Avg Rating',
                             title="Author Popularity vs. Rating")
            st.plotly_chart(fig, use_container_width=True, key="author_popularity_vs_rating")

        elif question == "Recommend top Sci-Fi books":
            sci_fi = df[df['Genre'].str.contains('Science Fiction', case=False, na=False)]
            top_sci_fi = sci_fi.sort_values(by='Rating', ascending=False).head(5)
            st.dataframe(top_sci_fi[['Book Name', 'Author', 'Genre', 'Rating']])

        else:  # Recommend for Thriller lovers
            thriller = df[df['Genre'].str.contains('Thriller', case=False, na=False)]
            top_thriller = thriller.sort_values(by='Rating', ascending=False).head(5)
            st.dataframe(top_thriller[['Book Name', 'Author', 'Genre', 'Rating']])

# --- Recommendations Page ---
elif page == "Recommendations":
    st.title("🎯 Personalized Recommendations")
    st.markdown("""
    ### Tailored Audiobook Suggestions
    Choose your preferences and discover the best books suited to your taste.
    ---
    """)

    # Filter Selection
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Choose filter type", ["Genre", "Author"])
    with col2:
        method = st.selectbox("Recommendation method", ["Content-Based", "Cluster-Based", "Hybrid (Genre + Content-Based)"])

    # Fixed number of recommendations
    top_n = 5  # Fixed value for number of recommendations

    # Value Selection based on filter
    if filter_type == "Genre":
        selected_value = st.selectbox("Select Genre", sorted(df["Genre"].dropna().unique()))
        filtered_df = df[df["Genre"] == selected_value]
    else:
        selected_value = st.selectbox("Select Author", sorted(df["Author"].dropna().unique()))
        filtered_df = df[df["Author"] == selected_value]

    st.markdown(f"---\n### 🔎 Top {top_n} {method} Recommendations based on {filter_type}: `{selected_value}`")

    # --- Recommendation Functions ---
    def content_based_recommendation(dataframe):
        idx = dataframe.index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        book_indices = [i[0] for i in sim_scores if i[0] != idx][:top_n]
        return df.iloc[book_indices]

    def cluster_based_recommendation(dataframe):
        cluster_id = dataframe["cluster"].values[0]
        cluster_books = df[df["cluster"] == cluster_id].sort_values(by="Rating", ascending=False)
        return cluster_books.head(top_n)

    def hybrid_recommendation(dataframe):
        genre = dataframe["Genre"].values[0]
        idx = dataframe.index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        filtered_indices = [i[0] for i in sim_scores if df.iloc[i[0]]["Genre"] == genre and i[0] != idx][:top_n]
        return df.iloc[filtered_indices]

    # --- Display Results ---
    if not filtered_df.empty:
        if method == "Content-Based":
            results = content_based_recommendation(filtered_df)
        elif method == "Cluster-Based":
            results = cluster_based_recommendation(filtered_df)
        else:
            results = hybrid_recommendation(filtered_df)

        for _, row in results.iterrows():
            st.markdown(f"""
            <div class="book-card">
                <h4>{row['Book Name']}</h4>
                <p><strong>Author:</strong> {row['Author']}</p>
                <p><strong>Genre:</strong> {row['Genre']}</p>
                <p><strong>Rating:</strong> {row['Rating']} ⭐</p>
                <p><strong>Reviews:</strong> {row['Number of Reviews']}</p>
                <p><strong>Listening Time:</strong> {row['Listening Time (minutes)']} mins</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No data available for the selected input.")

# --- Contact Page ---
elif page == "Contact":
    st.title("📬 Get in Touch")
    st.markdown("""
    ### We'd love to hear from you!
    Whether it's feedback, collaboration ideas, or just a hello—reach out anytime.
    ---

    📧 **Email:** [priyasmalavade@gmail.com](mailto:priyasmalavade@gmail.com)

    💼 **LinkedIn:** [Connect with me](https://www.linkedin.com/in/priyanka-malavade-b34677298/)

    
    """)