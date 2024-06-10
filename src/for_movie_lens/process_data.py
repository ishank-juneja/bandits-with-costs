import argparse
import numpy as np
import pandas as pd


# Standard argparse spiel here to get the two CSV files, and hard-code in their default paths/values
parser = argparse.ArgumentParser()
parser.add_argument("--ratings-file", action="store", dest="ratings", default="data/ml-25m/ratings.csv")
parser.add_argument("--movies-file", action="store", dest="movies", default="data/ml-25m/movies.csv")

# Parse the arguments
args = parser.parse_args()
# Load the CSV files
ratings = pd.read_csv(args.ratings)
movies = pd.read_csv(args.movies)

# Merge the dataframes on movieId
merged_df = pd.merge(ratings, movies, on='movieId')

# Split genres into separate rows
merged_df['genres'] = merged_df['genres'].str.split('|')
exploded_df = merged_df.explode('genres')

# Calculate average rating by genre
genre_avg_rating = exploded_df.groupby('genres')['rating'].mean().reset_index()

# Rename columns for clarity
genre_avg_rating.columns = ['genre', 'avg-rating']

# Rescale the average-rating to be a number between 0 and 1 by dividing by 5
genre_avg_rating['avg-rating'] = genre_avg_rating['avg-rating'] / 5

# Display the resulting DataFrame
# print(genre_avg_rating)

# Retrieve the number of genres
num_genres = len(genre_avg_rating)

np.random.seed(0)
# Create a numpy array of random numbers between 0 and 1 of length equal to the number of genres
rand_costs = np.random.rand(num_genres)

# Create a new column in the DataFrame for the costs
genre_avg_rating['cost'] = rand_costs

# Sort the DataFrame in the ascending order of the costs
genre_avg_rating = genre_avg_rating.sort_values('cost', ascending=True)

# Save the bandit instance constructed from these values as a text file in the below format
# instance_id: FCS001
# arm_reward_array: 0.4, 0.5, 0.6
# subsidy_factor : 0.2
# arm_cost_array: 0.15, 0.2, 0.25

# Open a file to write the bandit instance
with open('data/ml-25m/bandit_instance.txt', 'w') as f:
    f.write('instance_id: ML001\n')
    # Format rewardsand costs to 3 decimal places
    f.write('arm_reward_array: ' + ', '.join(genre_avg_rating['avg-rating'].apply(lambda x: f'{x:.3f}').values) + '\n')
    f.write('subsidy_factor: 0.2\n')
    f.write('arm_cost_array: ' + ', '.join(genre_avg_rating['cost'].apply(lambda x: f'{x:.3f}').values) + '\n')
