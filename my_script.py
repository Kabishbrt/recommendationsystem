import os
import json
import sys
import logging
import numpy as np
import pandas as pd

bookpath= os.path.join(os.getcwd(),"RecommendationSystem", "Books.csv")
userpath = os.path.join(os.getcwd(),"RecommendationSystem", "Users.csv")
ratingspath= os.path.join(os.getcwd(),"RecommendationSystem", "Ratings.csv")
books = pd.read_csv(bookpath,low_memory=False)
users = pd.read_csv(userpath)
ratings= pd.read_csv(ratingspath)
# users = pd.read_csv(r".\RecommendationSystem\users.csv")
# ratings = pd.read_csv(r".\RecommendationSystem\ratings.csv")

# Manually calculate cosine similarity
def cosine_similarity_manual(matrix):
    """
    Calculate the cosine similarity between rows of a matrix.

    Parameters:
    - matrix: 2D numpy array where each row represents a vector.

    Returns:
    - similarity_scores: 2D numpy array containing cosine similarity scores.
    """
    # Manually calculate dot product
    dot_product = np.array([[np.sum(matrix[i] * matrix[j]) for j in range(matrix.shape[0])] for i in range(matrix.shape[0])])

    # Manually calculate normalization
    norm = np.sqrt(np.sum(matrix**2, axis=1))
    norm_matrix = np.outer(norm, norm)
    
    # Calculate cosine similarity scores
    similarity_scores = dot_product / norm_matrix
    
    return similarity_scores

# Filtered DataFrame based on user ratings
ratings_with_name = ratings.merge(books,on='ISBN')
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] >200
criticalusers = x[x].index

filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(criticalusers)]

# Filtered DataFrame based on famous books
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index

final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# Create a pivot table
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

pt.fillna(0, inplace=True)

# Manually calculate cosine similarity
similarity_scores_manual = cosine_similarity_manual(pt.values)

# Recommendation function
def recommend(book_name):
    try:
        # index fetch
        index = np.where(pt.index == book_name)[0][0]
        #print("Found index: {index}")
        
        similar_items = sorted(list(enumerate(similarity_scores_manual[index])), key=lambda x: x[1], reverse=True)[1:5]
        
        data = []
        for i in similar_items:
            item = {}
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item['BookTitle'] = temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]
            #item['ISBN'] = temp_df.drop_duplicates('Book-Title')['ISBN'].values[0]  # Assuming ISBN is a column in your dataset
            #item['ImageURL'] = temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0]
        
            data.append(item)
        return data
    except IndexError:
        print(f"Book '{book_name}' not found in the dataset.")
        return None
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None


#below is the code for running python script through cmd.. ex: python my_script.py "1984"

if __name__ == "__main__":
    try:
        # Extract command-line arguments
        book_title = sys.argv[1]

        # Call the function with the command-line argument
        data = recommend(book_title)

        # Convert the data array to a JSON string
        json_data = json.dumps(data)

        # Print the JSON data to the standard output
        print(json_data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)