from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load data
df = pd.read_csv('sgp1.csv')

# Text preprocessing
nltk.download('stopwords')
stemmer = PorterStemmer()
stopwords_eng = set(stopwords.words('english'))


def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        words = [stemmer.stem(word) for word in words if word not in stopwords_eng]
        return ' '.join(words)
    else:
        return '' 

df['text'] = df['jobdescription'] + ' ' + df['skills1']
df['text'] = df['text'].fillna('').apply(clean_text)
df['text'] = df['text'].apply(clean_text)

# Generate vectors
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(df['text'])

# Compute similarity
similarity = cosine_similarity(vectors)


# Recommendation function
def get_recommendations(job_desc, skills, top_n):
    cleaned_text = clean_text(job_desc + ' ' + skills)
    vector = tfidf.transform([cleaned_text])
    similarity_scores = cosine_similarity(vector, vectors).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices]['jobtitle'].tolist()
    return recommendations


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    job_desc = request.form['job_desc']
    skills = request.form['skills']
    top_n = int(request.form['top_n'])
    recommendations = get_recommendations(job_desc, skills, top_n)
    return render_template('index.html', job_desc=job_desc, skills=skills, top_n=top_n, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
