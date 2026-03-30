# 🎬 Movie Recommendation System
 
A content-based and popularity-driven movie recommendation engine built on the [IMDB Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), combining classical NLP techniques with IMDB's weighted rating formula to surface relevant film recommendations.
 
---
 
## Overview
 
This project implements two complementary recommendation strategies:
 
- **Popularity-based recommender** — ranks movies using the IMDB weighted rating formula, balancing average score against vote count to avoid obscure high-rated films dominating the chart.
- **Content-based recommender (TF-IDF)** — finds movies with similar plot descriptions using TF-IDF vectorization on movie overviews and cosine similarity.
- **Content-based recommender (Metadata Soup)** — enriches similarity signals by combining cast, director, genres, and keywords into a "soup" string, then applying Count Vectorization and cosine similarity.
 
---
 
## Dataset
 
**Source:** [The Movies Dataset — Kaggle (rounakbanik)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
 
| File | Description |
|---|---|
| `movies_metadata.csv` | Core metadata: title, overview, genres, vote count, vote average, budget, revenue |
| `credits.csv` | Full cast and crew per movie |
| `keywords.csv` | Plot keywords per movie |
 
- ~45,000 movies after cleaning
- 3 malformed rows removed (`id` values: 19730, 29503, 35587)
- Duplicates resolved by quality score (vote count, popularity, revenue, overview completeness)
 
---
 
## Methodology
 
### 1. Popularity-Based Recommender
 
Uses the **IMDB Weighted Rating Formula**:
 
```
score = (v / (v + M)) × R + (M / (M + v)) × C
```
 
Where:
- `v` = number of votes for the movie
- `M` = minimum votes required (90th percentile threshold)
- `R` = average rating of the movie
- `C` = mean vote across the entire dataset
 
Only movies above the 90th percentile vote count threshold qualify, ensuring statistical reliability.
 
**Top results include:** The Shawshank Redemption, The Godfather, The Dark Knight, Pulp Fiction, Schindler's List.
 
---
 
### 2. Content-Based Recommender — TF-IDF on Overviews
 
**Pipeline:**
1. Fill missing overviews with empty strings
2. Apply `TfidfVectorizer` (English stop words removed)
3. Compute pairwise cosine similarity via `linear_kernel`
4. Build a title→index reverse map
5. Return top-10 most similar movies by cosine score
 
**Key insight:** TF-IDF down-weights common words (e.g. "man", "world") and up-weights distinctive plot terms. Works well for movies with descriptive, unique overviews — but fails when overviews are generic or sparse.
 
---
 
### 3. Content-Based Recommender — Metadata Soup
 
**Pipeline:**
1. Parse `cast`, `crew`, `keywords`, `genres` from stringified JSON
2. Extract director from crew; keep top-3 cast members
3. Lowercase and strip spaces from all name tokens (e.g. `"Francis Ford Coppola"` → `"francisfordcoppola"`)
4. Concatenate into a single "soup" string per movie
5. Apply `CountVectorizer` and compute cosine similarity
 
**Why lowercase + strip spaces?** Prevents "Al Pacino" from matching any single word "al" or "pacino" that might appear in unrelated contexts. Each person/keyword becomes a unique atomic token.
 
**Sample soup for The Godfather:**
```
godfather mafia murder organized crime marlon brando al pacino james caan francisfordcoppola drama crime
```
 
---
 
## Data Cleaning
 
| Issue | Resolution |
|---|---|
| 3 rows with non-numeric IDs | Dropped before type conversion |
| Duplicate movie IDs | Resolved by composite quality score; best row kept |
| Mixed-type columns | `pd.to_numeric(..., errors='coerce')` applied |
| Missing overviews | Filled with empty string before vectorization |
| Index misalignment after sorting | `reset_index(drop=True)` applied before building similarity matrix |
 
> **Important:** The index must be reset *after* all sorting and deduplication, and *before* building the vectorizer matrix. Failure to do so causes the title→row lookup to map to wrong movies in the similarity matrix.
 
---
 
## Project Structure
 
```
movie-recommender/
│
├── data/
│   ├── movies_metadata.csv
│   ├── credits.csv
│   └── keywords.csv
│
├── notebooks/
│   ├── Recommender System IMDB.ipynb
│
│
└── README.md
```
 
---
 
## Dependencies
 
```
pandas
numpy
scikit-learn
nltk
ast
```
 
Install with:
```bash
pip install pandas numpy scikit-learn nltk
```
 
---
 
## Usage
 
```python
# Popularity-based top 15
print(qualified_movies[['title', 'vote_count', 'vote_average', 'score']].head(15))
 
# Content-based (TF-IDF overview)
get_recommendations('The Dark Knight Rises', cosine_sim_Overviews, indices_tfidf)
 
# Content-based (metadata soup)
get_recommendations('The Godfather', cosine_sim_soup, indices_soup)
```
 
---
 
## Sample Results
 
### Top Rated (Popularity Model)
| Title | Votes | Avg Rating | Weighted Score |
|---|---|---|---|
| The Shawshank Redemption | 8358 | 8.5 | 8.446 |
| The Godfather | 6024 | 8.5 | 8.425 |
| The Dark Knight | 12269 | 8.3 | 8.265 |
| Pulp Fiction | 8670 | 8.3 | 8.251 |
 
### Similar to "The Godfather" (Soup Model)
1. The Godfather: Part III
2. The Godfather: Part II
3. The Rain People
4. Rege
5. Last Exit
 
---
 
## Key Learnings
 
- **TF-IDF vs Count Vectorizer:** TF-IDF is better for free-text overviews (penalises common words); Count Vectorizer suits the soup approach where every token is already a meaningful atomic name.
- **Index alignment is critical:** Sorting or deduplicating after building the similarity matrix — or before calling `reset_index()` — causes a silent mismatch where the wrong movie is returned.
- **Deduplication strategy matters:** Simply dropping duplicates on ID loses data. Scoring rows by completeness and keeping the richest version produces a cleaner dataset.
- **Token normalisation:** Merging first-last names into a single string prevents false partial matches and makes person-based similarity more reliable.
 
---
 
## Future Work
 
- [ ] Collaborative filtering using user ratings (`ratings.csv`)
- [ ] Genre-filtered recommendations
- [ ] Web interface (Streamlit or FastAPI)
- [ ] Embedding-based similarity using sentence transformers
 
---
 
## License
 
MIT License. Dataset sourced from Kaggle under the original license terms of the TMDB Movies Dataset.
