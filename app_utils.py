import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_recommendation_chart(recommendations):
    """Create a chart showing recommendation scores"""
    if not recommendations:
        return None
    
    book_names = [rec['book_name'][:30] + '...' if len(rec['book_name']) > 30 else rec['book_name'] 
                  for rec in recommendations[:10]]
    ratings = [rec['rating'] for rec in recommendations[:10]]
    scores = [rec.get('similarity_score', rec.get('total_score', 0)) for rec in recommendations[:10]]
    
    fig = go.Figure()
    
    # Add rating bars
    fig.add_trace(go.Bar(
        name='Rating',
        x=book_names,
        y=ratings,
        yaxis='y',
        offsetgroup=1,
        marker_color='lightblue'
    ))
    
    # Add similarity score line
    fig.add_trace(go.Scatter(
        name='Similarity Score',
        x=book_names,
        y=scores,
        yaxis='y2',
        mode='lines+markers',
        marker_color='red',
        line=dict(width=3)
    ))
    
    # Update layout
    fig.update_layout(
        title='Recommendation Analysis',
        xaxis_title='Books',
        yaxis=dict(title='Rating', side='left'),
        yaxis2=dict(title='Similarity Score', side='right', overlaying='y'),
        barmode='group',
        height=500
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def export_recommendations_csv(recommendations):
    """Convert recommendations to CSV format for export"""
    if not recommendations:
        return None
    
    export_data = []
    for i, rec in enumerate(recommendations, 1):
        export_data.append({
            'Rank': i,
            'Book Name': rec['book_name'],
            'Author': rec['author'],
            'Rating': rec['rating'],
            'Number of Reviews': rec['num_reviews'],
            'Price ($)': rec['price'],
            'Genres': ', '.join(rec['genres'][:5]),
            'Similarity Score': rec.get('similarity_score', rec.get('total_score', 0))
        })
    
    return pd.DataFrame(export_data)

def create_genre_distribution_chart(df):
    """Create a pie chart showing genre distribution"""
    all_genres = []
    for genre_list in df['Genres']:
        if isinstance(genre_list, list):
            all_genres.extend(genre_list)
    
    from collections import Counter
    genre_counts = Counter(all_genres)
    top_genres = dict(genre_counts.most_common(10))
    
    fig = px.pie(
        values=list(top_genres.values()),
        names=list(top_genres.keys()),
        title='Top 10 Genres Distribution'
    )
    
    return fig

def format_price(price):
    """Format price for display"""
    if price < 1000:
        return f"${price:.2f}"
    else:
        return f"${price/100:.2f}"  # Assuming prices are in cents

def get_book_statistics(df):
    """Get comprehensive book statistics"""
    stats = {
        'total_books': len(df),
        'unique_authors': df['Author'].nunique(),
        'avg_rating': df['Rating'].mean(),
        'median_rating': df['Rating'].median(),
        'avg_price': df['Price'].mean(),
        'median_price': df['Price'].median(),
        'avg_reviews': df['Number_of_Reviews'].mean(),
        'total_reviews': df['Number_of_Reviews'].sum(),
        'rating_distribution': df['Rating_Category'].value_counts().to_dict() if 'Rating_Category' in df.columns else {},
        'price_distribution': df['Price_Category'].value_counts().to_dict() if 'Price_Category' in df.columns else {}
    }
    
    return stats

def search_books(df, query, search_type='all'):
    """Advanced book search functionality"""
    query = query.lower().strip()
    
    if not query:
        return pd.DataFrame()
    
    if search_type == 'title':
        mask = df['Book_Name'].str.lower().str.contains(query, na=False)
    elif search_type == 'author':
        mask = df['Author'].str.lower().str.contains(query, na=False)
    elif search_type == 'genre':
        mask = df['Genres'].apply(
            lambda genres: any(query in genre.lower() for genre in genres) 
            if isinstance(genres, list) else False
        )
    else:  # search_type == 'all'
        title_mask = df['Book_Name'].str.lower().str.contains(query, na=False)
        author_mask = df['Author'].str.lower().str.contains(query, na=False)
        genre_mask = df['Genres'].apply(
            lambda genres: any(query in genre.lower() for genre in genres) 
            if isinstance(genres, list) else False
        )
        mask = title_mask | author_mask | genre_mask
    
    results = df[mask].copy()
    
    # Sort by relevance (rating * log(reviews + 1))
    results['relevance_score'] = results['Rating'] * np.log1p(results['Number_of_Reviews'])
    results = results.sort_values('relevance_score', ascending=False)
    
    return results.head(20)  # Return top 20 results

def validate_user_input(preferences):
    """Validate user input for recommendations"""
    errors = []
    
    if not preferences.get('genres'):
        errors.append("Please select at least one genre")
    
    if preferences.get('min_rating', 0) < 1 or preferences.get('min_rating', 0) > 5:
        errors.append("Rating must be between 1 and 5")
    
    if preferences.get('max_price', 0) < 0:
        errors.append("Price must be positive")
    
    return errors

def create_user_preference_summary(preferences):
    """Create a summary of user preferences"""
    summary = []
    
    if preferences.get('genres'):
        summary.append(f"Genres: {', '.join(preferences['genres'][:3])}")
    
    if preferences.get('min_rating'):
        summary.append(f"Min Rating: {preferences['min_rating']}")
    
    if preferences.get('max_price'):
        summary.append(f"Max Price: ${preferences['max_price']}")
    
    if preferences.get('length_category'):
        summary.append(f"Length: {preferences['length_category']}")
    
    return " | ".join(summary)

def create_advanced_search_interface(df):
    """Create an advanced search interface for the Streamlit app"""
    st.subheader("🔍 Advanced Search")
    
    # Search options
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_query = st.text_input("Search for books:", placeholder="Enter book title, author, or keyword")
    
    with col2:
        search_type = st.selectbox(
            "Search in:",
            ["All", "Title", "Author", "Genre"],
            index=0
        )
    
    if search_query:
        # Perform search
        search_type_map = {
            "All": "all",
            "Title": "title", 
            "Author": "author",
            "Genre": "genre"
        }
        
        results = search_books(df, search_query, search_type_map[search_type])
        
        if not results.empty:
            st.success(f"Found {len(results)} results for '{search_query}'")
            
            # Display results
            for idx, book in results.iterrows():
                with st.expander(f"📚 {book['Book_Name']} by {book['Author']} ({book['Rating']}⭐)"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.metric("Rating", f"{book['Rating']}⭐")
                        st.metric("Reviews", f"{book['Number_of_Reviews']:,}")
                        st.metric("Price", f"${book['Price']:.2f}")
                    
                    with col2:
                        if isinstance(book['Genres'], list):
                            st.write(f"**Genres:** {', '.join(book['Genres'][:5])}")
                        else:
                            st.write("**Genres:** N/A")
                        
                        if 'Description_Clean' in book and pd.notna(book['Description_Clean']):
                            st.write(f"**Description:** {book['Description_Clean'][:200]}...")
        else:
            st.warning(f"No results found for '{search_query}' in {search_type.lower()}")
    
    return search_query

def display_book_metrics(df):
    """Display key metrics about the book dataset"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📚 Total Books",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="✍️ Authors",
            value=f"{df['Author'].nunique():,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="⭐ Avg Rating",
            value=f"{df['Rating'].mean():.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="💰 Avg Price",
            value=f"${df['Price'].mean():.0f}",
            delta=None
        )

def create_rating_filter(df):
    """Create a rating filter widget"""
    min_rating, max_rating = st.slider(
        "Filter by Rating Range:",
        min_value=float(df['Rating'].min()),
        max_value=float(df['Rating'].max()),
        value=(float(df['Rating'].min()), float(df['Rating'].max())),
        step=0.1,
        format="%.1f"
    )
    return min_rating, max_rating

def create_price_filter(df):
    """Create a price filter widget"""
    min_price, max_price = st.slider(
        "Filter by Price Range ($):",
        min_value=float(df['Price'].min()),
        max_value=float(df['Price'].max()),
        value=(float(df['Price'].min()), float(df['Price'].max())),
        step=50.0,
        format="$%.0f"
    )
    return min_price, max_price

def filter_dataframe(df, min_rating=None, max_rating=None, min_price=None, max_price=None, selected_genres=None):
    """Filter dataframe based on various criteria"""
    filtered_df = df.copy()
    
    # Apply rating filter
    if min_rating is not None and max_rating is not None:
        filtered_df = filtered_df[
            (filtered_df['Rating'] >= min_rating) & 
            (filtered_df['Rating'] <= max_rating)
        ]
    
    # Apply price filter
    if min_price is not None and max_price is not None:
        filtered_df = filtered_df[
            (filtered_df['Price'] >= min_price) & 
            (filtered_df['Price'] <= max_price)
        ]
    
    # Apply genre filter
    if selected_genres:
        genre_mask = filtered_df['Genres'].apply(
            lambda genres: any(genre in selected_genres for genre in genres) 
            if isinstance(genres, list) else False
        )
        filtered_df = filtered_df[genre_mask]
    
    return filtered_df