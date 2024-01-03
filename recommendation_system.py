from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Sample dataset (replace this with your actual dataset)
data = pd.DataFrame({
    'ItemID': [1, 2, 3, 4],
    'ItemName': ['Item A', 'Item B', 'Item C', 'Item D'],
    'Description': ['Description A', 'Description B', 'Description C', 'Description D']
})

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Description'].fillna(''))

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Routes
@app.route('/')
def index():
    return render_template('index.html', items=data['ItemName'])

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_item = request.form['selected_item']
    item_index = data[data['ItemName'] == selected_item].index[0]

    # Get the pairwsie similarity scores of all items with the selected item
    sim_scores = list(enumerate(cosine_sim[item_index]))

    # Sort the items based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar items
    sim_scores = sim_scores[1:6]

    # Get the item indices
    item_indices = [score[0] for score in sim_scores]

    # Return the top 5 recommended items
    recommended_items = data['ItemName'].iloc[item_indices].tolist()

    return render_template('recommendation.html', selected_item=selected_item, recommended_items=recommended_items)

if __name__ == '__main__':
    app.run(debug=True)
