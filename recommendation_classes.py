# recommendation_classes.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

class ContentBasedRecommender:
    """Content-based recommendation system using TF-IDF similarities"""
    
    def __init__(self, df, tfidf_matrix, book_to_idx):
        self.df = df
        self.tfidf_matrix = tfidf_matrix
        self.book_to_idx = book_to_idx
        self.similarity_matrix = None
        self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self):
        """Compute cosine similarity matrix for all books"""
        print("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        print(f"Similarity matrix computed: {self.similarity_matrix.shape}")
    
    def get_recommendations(self, book_title, n_recommendations=10, min_rating=3.5):
        """Get content-based recommendations for a given book"""
        if book_title not in self.book_to_idx:
            return None, f"Book '{book_title}' not found in database"
        
        # Get book index
        book_idx = self.book_to_idx[book_title]
        
        # Get similarity scores for this book
        sim_scores = list(enumerate(self.similarity_matrix[book_idx]))
        
        # Sort books by similarity (excluding the book itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
        
        # Filter by minimum rating and get top recommendations
        recommendations = []
        for idx, score in sim_scores:
            book_data = self.df.iloc[idx]
            if book_data['Rating'] >= min_rating:
                recommendations.append({
                    'book_name': book_data['Book_Name'],
                    'author': book_data['Author'],
                    'rating': book_data['Rating'],
                    'num_reviews': book_data['Number_of_Reviews'],
                    'price': book_data['Price'],
                    'genres': book_data['Genres'],
                    'similarity_score': score
                })
                
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations, None
    
    def get_genre_based_recommendations(self, preferred_genres, n_recommendations=10, min_rating=4.0):
        """Get recommendations based on preferred genres"""
        genre_scores = {}
        
        for idx, row in self.df.iterrows():
            if row['Rating'] < min_rating:
                continue
                
            book_genres = row['Genres'] if isinstance(row['Genres'], list) else []
            
            # Calculate genre match score
            genre_matches = len(set(preferred_genres).intersection(set(book_genres)))
            if genre_matches > 0:
                # Weight by rating and number of matches
                score = genre_matches * row['Rating'] * np.log1p(row['Number_of_Reviews'])
                genre_scores[idx] = score
        
        # Sort by score and get top recommendations
        top_indices = sorted(genre_scores.keys(), key=lambda x: genre_scores[x], reverse=True)
        
        recommendations = []
        for idx in top_indices[:n_recommendations]:
            book_data = self.df.iloc[idx]
            recommendations.append({
                'book_name': book_data['Book_Name'],
                'author': book_data['Author'],
                'rating': book_data['Rating'],
                'num_reviews': book_data['Number_of_Reviews'],
                'price': book_data['Price'],
                'genres': book_data['Genres'],
                'genre_score': genre_scores[idx]
            })
        
        return recommendations


class ClusterBasedRecommender:
    """Recommendation system based on book clusters"""
    
    def __init__(self, df):
        self.df = df
        self.cluster_stats = self._compute_cluster_stats()
    
    def _compute_cluster_stats(self):
        """Compute statistics for each cluster"""
        stats = {}
        for cluster_id in self.df['Cluster_KMeans'].unique():
            cluster_books = self.df[self.df['Cluster_KMeans'] == cluster_id]
            
            # Get cluster genres
            cluster_genres = []
            for genre_list in cluster_books['Genres']:
                if isinstance(genre_list, list):
                    cluster_genres.extend(genre_list)
            
            top_genres = dict(Counter(cluster_genres).most_common(5))
            
            stats[cluster_id] = {
                'size': len(cluster_books),
                'avg_rating': cluster_books['Rating'].mean(),
                'avg_price': cluster_books['Price'].mean(),
                'avg_reviews': cluster_books['Number_of_Reviews'].mean(),
                'top_genres': list(top_genres.keys())
            }
        
        return stats
    
    def get_cluster_recommendations(self, book_title, n_recommendations=10, exclude_input=True):
        """Get recommendations from the same cluster as the input book"""
        # Create book_to_idx mapping
        book_to_idx = {book: idx for idx, book in enumerate(self.df['Book_Name'])}
        
        if book_title not in book_to_idx:
            return None, f"Book '{book_title}' not found in database"
        
        # Find the book's cluster
        book_idx = book_to_idx[book_title]
        book_cluster = self.df.iloc[book_idx]['Cluster_KMeans']
        
        # Get all books in the same cluster
        cluster_books = self.df[self.df['Cluster_KMeans'] == book_cluster].copy()
        
        # Exclude the input book if requested
        if exclude_input:
            cluster_books = cluster_books[cluster_books['Book_Name'] != book_title]
        
        # Create a popularity score using log of reviews + 1
        cluster_books['popularity_score'] = np.log1p(cluster_books['Number_of_Reviews'])
        
        # Sort by rating and popularity
        cluster_books['combined_score'] = (
            cluster_books['Rating'] * 0.6 + 
            (cluster_books['popularity_score'] / cluster_books['popularity_score'].max()) * 0.4
        )
        
        top_books = cluster_books.nlargest(n_recommendations, 'combined_score')
        
        recommendations = []
        for idx, row in top_books.iterrows():
            recommendations.append({
                'book_name': row['Book_Name'],
                'author': row['Author'],
                'rating': row['Rating'],
                'num_reviews': row['Number_of_Reviews'],
                'price': row['Price'],
                'genres': row['Genres'],
                'cluster_id': row['Cluster_KMeans'],
                'combined_score': row['combined_score']
            })
        
        cluster_info = self.cluster_stats[book_cluster]
        return recommendations, cluster_info
    
    def get_diverse_recommendations(self, n_per_cluster=2, min_rating=4.0):
        """Get diverse recommendations by selecting top books from each cluster"""
        recommendations = []
        
        for cluster_id in sorted(self.df['Cluster_KMeans'].unique()):
            cluster_books = self.df[
                (self.df['Cluster_KMeans'] == cluster_id) & 
                (self.df['Rating'] >= min_rating)
            ].copy()
            
            if len(cluster_books) == 0:
                continue
            
            # Create popularity score using log of reviews + 1
            cluster_books['popularity_score'] = np.log1p(cluster_books['Number_of_Reviews'])
            
            # Sort by rating and popularity
            cluster_books['score'] = (
                cluster_books['Rating'] * 0.7 + 
                (cluster_books['popularity_score'] / cluster_books['popularity_score'].max()) * 0.3
            )
            
            top_cluster_books = cluster_books.nlargest(n_per_cluster, 'score')
            
            for idx, row in top_cluster_books.iterrows():
                recommendations.append({
                    'book_name': row['Book_Name'],
                    'author': row['Author'],
                    'rating': row['Rating'],
                    'num_reviews': row['Number_of_Reviews'],
                    'price': row['Price'],
                    'genres': row['Genres'],
                    'cluster_id': row['Cluster_KMeans'],
                    'cluster_genres': self.cluster_stats[cluster_id]['top_genres']
                })
        
        return recommendations


class HybridRecommender:
    """Hybrid recommendation system combining multiple approaches"""
    
    def __init__(self, content_rec, cluster_rec, df):
        self.content_rec = content_rec
        self.cluster_rec = cluster_rec
        self.df = df
    
    def get_hybrid_recommendations(self, book_title, n_recommendations=10, 
                                 content_weight=0.6, cluster_weight=0.4):
        """Combine content-based and cluster-based recommendations"""
        
        # Get content-based recommendations
        content_recs, error = self.content_rec.get_recommendations(
            book_title, n_recommendations=20, min_rating=3.5
        )
        if error:
            return None, error
        
        # Get cluster-based recommendations
        cluster_recs, cluster_info = self.cluster_rec.get_cluster_recommendations(
            book_title, n_recommendations=20
        )
        
        if not cluster_recs:
            # If cluster recommendations fail, return content-based only
            return content_recs[:n_recommendations], None
        
        # Combine and score recommendations
        all_recommendations = {}
        
        # Add content-based scores
        for i, rec in enumerate(content_recs):
            book_name = rec['book_name']
            content_score = rec['similarity_score'] * content_weight
            # Higher rank gets higher score
            rank_bonus = (len(content_recs) - i) / len(content_recs) * 0.1
            
            all_recommendations[book_name] = {
                **rec,
                'content_score': content_score,
                'cluster_score': 0,
                'total_score': content_score + rank_bonus,
                'source': 'content'
            }
        
        # Add cluster-based scores
        for i, rec in enumerate(cluster_recs):
            book_name = rec['book_name']
            cluster_score = rec['combined_score'] * cluster_weight
            rank_bonus = (len(cluster_recs) - i) / len(cluster_recs) * 0.1
            
            if book_name in all_recommendations:
                # Book appears in both systems - combine scores
                all_recommendations[book_name]['cluster_score'] = cluster_score
                all_recommendations[book_name]['total_score'] += cluster_score + rank_bonus
                all_recommendations[book_name]['source'] = 'hybrid'
            else:
                # Book only in cluster recommendations
                all_recommendations[book_name] = {
                    **rec,
                    'content_score': 0,
                    'cluster_score': cluster_score,
                    'total_score': cluster_score + rank_bonus,
                    'source': 'cluster'
                }
        
        # Sort by total score and return top recommendations
        sorted_recs = sorted(all_recommendations.values(), 
                           key=lambda x: x['total_score'], reverse=True)
        
        return sorted_recs[:n_recommendations], cluster_info
    
    def get_personalized_recommendations(self, user_preferences, n_recommendations=10):
        """Get personalized recommendations based on user preferences"""
        
        preferred_genres = user_preferences.get('genres', [])
        min_rating = user_preferences.get('min_rating', 4.0)
        max_price = user_preferences.get('max_price', float('inf'))
        preferred_length = user_preferences.get('length_category', None)
        
        # Filter books based on preferences
        filtered_df = self.df[
            (self.df['Rating'] >= min_rating) & 
            (self.df['Price'] <= max_price)
        ].copy()
        
        if preferred_length and 'Length_Category' in self.df.columns:
            filtered_df = filtered_df[filtered_df['Length_Category'] == preferred_length]
        
        # Create popularity score
        filtered_df['popularity_score'] = np.log1p(filtered_df['Number_of_Reviews'])
        
        # Calculate personalized scores
        recommendations = []
        for idx, row in filtered_df.iterrows():
            book_genres = row['Genres'] if isinstance(row['Genres'], list) else []
            
            # Genre match score
            genre_matches = len(set(preferred_genres).intersection(set(book_genres)))
            genre_score = genre_matches / max(len(preferred_genres), 1) if preferred_genres else 0.5
            
            # Combined score
            total_score = (
                row['Rating'] * 0.4 +
                genre_score * 0.3 +
                (row['popularity_score'] / filtered_df['popularity_score'].max()) * 0.2 +
                (1 - row['Price'] / self.df['Price'].max()) * 0.1
            )
            
            recommendations.append({
                'book_name': row['Book_Name'],
                'author': row['Author'],
                'rating': row['Rating'],
                'num_reviews': row['Number_of_Reviews'],
                'price': row['Price'],
                'genres': row['Genres'],
                'genre_match_score': genre_score,
                'total_score': total_score
            })
        
        # Sort by total score
        recommendations.sort(key=lambda x: x['total_score'], reverse=True)
        return recommendations[:n_recommendations]