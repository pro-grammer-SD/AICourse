import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'title': [
        'The Matrix', 'John Wick', 'Inception', 'Interstellar', 'The Dark Knight',
        'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'The Social Network', 'Tenet'
    ],
    'description': [
        'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
        'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.',
        'A thief who steals corporate secrets through dream-sharing technology is given a task of inception.',
        'A group of explorers travel through a wormhole in space in an attempt to ensure humanity’s survival.',
        'Batman faces the Joker, a criminal mastermind who plunges Gotham into chaos.',
        'The lives of two mob hitmen, a boxer, and others intertwine in a series of unexpected events.',
        'An insomniac office worker forms an underground fight club with a soap maker.',
        'The life story of Forrest Gump, a man with a low IQ who witnesses and influences historical events.',
        'A story about the creation of Facebook and the lawsuits that followed.',
        'A secret agent embarks on a time-bending mission to prevent World War III.'
    ]
}

df = pd.DataFrame(data)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

print(recommend('Inception'))
