import pandas as pd

# Load the CSV files
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Merge the dataframes on movieId
merged_df = pd.merge(ratings, movies, on='movieId')

# Split genres into separate rows
merged_df['genres'] = merged_df['genres'].str.split('|')
exploded_df = merged_df.explode('genres')

# Calculate average rating by genre
genre_avg_rating = exploded_df.groupby('genres')['rating'].mean().reset_index()

# Rename columns for clarity
genre_avg_rating.columns = ['genre', 'avg-rating']

# Display the resulting DataFrame
print(genre_avg_rating)
