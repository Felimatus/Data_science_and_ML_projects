"""
hybrid_model.py -- self-contained loader + the full book-recommendation hybrid.

Loads the saved artifacts (models/) + raw data (cleaned_data/) once at import,
rebuilds the cheap structures the 5 base models need, then exposes the hybrid:

    from hybrid_model import load_recommender
    rec = load_recommender()
    rec("Lord of the Rings")            # -> DataFrame[ISBN, Title]

or directly:  from hybrid_model import hybrid_recommend; hybrid_recommend("...")
"""
import os
import io
import contextlib
import numpy as np
import pandas as pd
import re
import joblib
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.environ.get("REC_DATA_DIR", "cleaned_data")
MODEL_DIR = os.environ.get("REC_MODEL_DIR", "models")

# ---------- Load saved artifacts + raw data (runs once at import) ----------
books = pd.read_csv(f"{DATA_DIR}/Books.csv", dtype={"Year": "Int32", "TotalRating": "Int32"})
if "Language" not in books.columns:
    books["Language"] = "unknown"
books["Language"] = books["Language"].fillna("unknown")
ratings = pd.read_csv(f"{DATA_DIR}/Ratings.csv", dtype={"Rating": "Int32"})
# Mean rating per book (1-10); used as a display-only quality filter in the recommenders
_rating_sum = ratings.groupby("ISBN")["Rating"].sum()
books["Rel_Cum_Rating"] = (books["ISBN"].map(_rating_sum) / books["TotalRating"]).round(2)

lgb_model = joblib.load(f"{MODEL_DIR}/lgbm_recommender.pkl")
_A = joblib.load(f"{MODEL_DIR}/lgbm_artifacts.joblib")
U_sigma = _A["U_sigma"]; Vt = _A["Vt"]; user_means = _A["user_means"]; lgb_u_cnt = _A["lgb_u_cnt"]
user_enc = _A["user_enc"]; isbn_enc = _A["isbn_enc"]; isbn_dec = _A["isbn_dec"]
tfidf_matrix = _A["tfidf_matrix"]; content_isbn_to_idx = _A["content_isbn_to_idx"]
isbn_to_author = _A["isbn_to_author"]
DEFAULT_LANGUAGES = {"English"}
rng_lgb = np.random.RandomState(42)

# ---------- Rebuild the cheap structures (matrices, KNN, content list) ----------
_rows = ratings["UserID"].map(user_enc).to_numpy()
_cols = ratings["ISBN"].map(isbn_enc).to_numpy()
_vals = ratings["Rating"].to_numpy().astype(np.float32)
user_item_sparse = csr_matrix((_vals, (_rows, _cols)), shape=(len(user_enc), len(isbn_enc)))
item_user_sparse = user_item_sparse.T.tocsr()
knn_model = NearestNeighbors(metric="cosine", algorithm="brute").fit(item_user_sparse)
content_isbn_list = [isbn for isbn, _ in sorted(content_isbn_to_idx.items(), key=lambda kv: kv[1])]
books_content = books.copy()

def lgb_features(user_idxs, item_idxs):
    uu = np.asarray(user_idxs); ii = np.asarray(item_idxs)
    svd_pred = user_means[uu] + np.einsum("ij,ji->i", U_sigma[uu], Vt[:, ii])
    base = np.column_stack([lgb_u_cnt[uu], user_means[uu], svd_pred])
    return np.hstack([base, U_sigma[uu], Vt[:, ii].T])

# ============================ helpers ============================

# --- Helper lookups ---
# --- Global recommendation defaults (single source of truth for all 6 models) ---
N_RECS = 10               # default number of recommendations each model returns
MIN_REL_CUM_RATING = 7    # min mean rating (1-10) a book needs to appear in results
isbn_to_author = books.set_index('ISBN')['Author'].to_dict()

def find_isbns_by_title(query, books_df=books):
    """Return all ISBNs whose title contains `query` (case-insensitive)."""
    mask = books_df['Title'].str.contains(query, case=False, na=False)
    return set(books_df.loc[mask, 'ISBN'])



# --- Language scoping for recommendation OUTPUT (models still train on ALL languages) ---
DEFAULT_LANGUAGES = {"English"}   # what every *_recommend returns unless `languages` is given

def allowed_isbns(books_df, min_ratings, languages=None):
    """ISBNs eligible to be RECOMMENDED: enough ratings AND in an allowed language.

    languages:
        None  -> DEFAULT_LANGUAGES (English)
        a name / list / set, e.g. "Spanish" or {"English", "German"}
        "all" -> disable the language filter (recommend in any language)
    """
    mask = books_df["TotalRating"] >= min_ratings
    if languages != "all":
        if languages is None:
            langs = set(DEFAULT_LANGUAGES)
        elif isinstance(languages, str):
            langs = {languages}
        else:
            langs = set(languages)
        mask &= books_df["Language"].isin(langs)
    return set(books_df.loc[mask, "ISBN"])

def filter_same_work(result_df, title_query, sim_threshold=0.26):
    """Remove books that are likely the same work (same author + high content similarity)."""
    query_isbns = find_isbns_by_title(title_query, books_content)
    query_idxs = [content_isbn_to_idx[isbn] for isbn in query_isbns if isbn in content_isbn_to_idx]
    if not query_idxs or result_df.empty:
        return result_df

    def normalize_author(name):
        return re.sub(r'[^a-z]', '', name.lower()) # removes both the space and the apostrophe

    query_authors = {normalize_author(isbn_to_author.get(isbn, '')) for isbn in query_isbns}
    query_vec = np.asarray(tfidf_matrix[query_idxs].mean(axis=0)) #mean vector 

    drop_isbns = set()
    for _, row in result_df.iterrows():
        isbn = row['ISBN']
        author = row.get('Author', '')
        if normalize_author(author) not in query_authors: 
            continue
        if isbn in content_isbn_to_idx:
            sim = cosine_similarity(query_vec, tfidf_matrix[content_isbn_to_idx[isbn]]).flatten()[0]
            if sim > sim_threshold:
                drop_isbns.add(isbn)

    filtered = result_df[~result_df['ISBN'].isin(drop_isbns)].reset_index(drop=True)
    if len(drop_isbns) > 0:
        print(f"  Filtered {len(drop_isbns)} same-work duplicates (same author + content sim > {sim_threshold})")
    return filtered

def normalize_scores(df):
    """Min-max normalize the Score column to [0, 1]."""
    if df.empty or df['Score'].max() == df['Score'].min():
        return df
    df = df.copy()
    df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min())
    return df


def cap_per_family(result_df, max_per_family=2, content_threshold=0.5, title_threshold=0.3):
    """Diversify a ranked result list by capping near-identical 'families' (e.g. the
    whole Harry Potter series) to their top `max_per_family` rows by score.

    A family = SAME author AND (high TF-IDF content similarity OR strong title-token
    overlap). So a *series* (HP1..HP7, near-identical appeal) collapses, while an
    author's *distinct* works (e.g. The Hobbit vs The Silmarillion) survive.
    Assumes result_df is already sorted best-first. Tune the two thresholds with the
    'HP must collapse to 2, the 3 Tolkien works must stay' test.
    """
    if result_df.empty:
        return result_df

    def _author(n):
        return re.sub(r'[^a-z]', '', str(n).lower())
    _stop = {'the', 'and', 'of', 'a', 'an', 'or', 'book', 'novel', 'vol', 'volume',
             'part', 'series', 'paperback', 'edition', 'chronicles'}
    def _title_toks(t):
        return {w for w in re.findall(r'[a-z]+', str(t).lower()) if len(w) > 2 and w not in _stop}

    kept_idx, anchors, counts = [], [], {}   # anchors: (author, content_idx, title_toks, family_rep)
    for idx, row in result_df.iterrows():
        a = _author(row.get('Author', ''))
        ci = content_isbn_to_idx.get(row['ISBN'])
        tt = _title_toks(row.get('Title', ''))
        rep = None
        for (a2, ci2, tt2, rep2) in anchors:
            if not a or a2 != a:                 # different (or missing) author -> different family
                continue
            csim = 0.0
            if ci is not None and ci2 is not None:
                csim = cosine_similarity(tfidf_matrix[ci], tfidf_matrix[ci2]).flatten()[0]
            tsim = len(tt & tt2) / len(tt | tt2) if (tt or tt2) else 0.0
            if csim > content_threshold or tsim > title_threshold:
                rep = rep2
                break
        if rep is None:                          # start a new family
            kept_idx.append(idx); counts[row['ISBN']] = 1
            anchors.append((a, ci, tt, row['ISBN']))
        elif counts[rep] < max_per_family:       # family still has room
            kept_idx.append(idx); counts[rep] += 1
            anchors.append((a, ci, tt, rep))
        # else: family already full -> drop this row

    dropped = len(result_df) - len(kept_idx)
    if dropped > 0:
        print(f"  cap_per_family: dropped {dropped} rows exceeding {max_per_family}/family")
    return result_df.loc[kept_idx].reset_index(drop=True)


# ============================ base models ============================

def item_based_cf(title_query, books_df=books, n_recs=N_RECS, min_ratings=10,
                  candidate_pool=500, shrinkage=10, min_rcr=MIN_REL_CUM_RATING, languages=None):
    """
    Item-Based CF using KNN (cosine) on the raw item-user sparse matrix.

    - Edition pooling: all editions of the queried work are merged into one
      work-level vector (each user counted once) before a single KNN query,
      which removes the spurious matches that low-support editions otherwise
      inject via the max-similarity merge.
    - Significance weighting: each similarity is scaled by
      min(shared_raters, shrinkage) / shrinkage, down-weighting matches backed
      by only a few shared users. Set shrinkage=0 to disable.
    """
    query_isbns = find_isbns_by_title(title_query, books_df)
    query_idxs = [isbn_enc[isbn] for isbn in query_isbns if isbn in isbn_enc]
    if not query_idxs:
        print(f"No books found matching '{title_query}' in the rating matrix")
        return pd.DataFrame()

    # Pool all editions into one work-level vector (binary: each user counts once)
    pooled = np.asarray((item_user_sparse[query_idxs] > 0).sum(axis=0)).flatten() # indexed by user
    pooled = (pooled > 0).astype(np.float64) #if True -> 1, False -> 0; binary component
    #pooled is a vector in user-space but it represents the query book
    n_raters = int(pooled.sum())
    print(f"Item-CF '{title_query}': pooled {len(query_idxs)} editions "
          f"-> {n_raters} distinct raters (shrinkage C={shrinkage})")
    pooled_sp = csr_matrix(pooled.reshape(1, -1)) # (n_users,) vector -> (1, n_users) matrix

    # Single KNN query on the pooled vector; pull a large pool before filtering
    n_req = min(item_user_sparse.shape[0], candidate_pool + len(query_idxs))
    distances, indices = knn_model.kneighbors(pooled_sp, n_neighbors=n_req) # nearer books

    query_set = set(query_idxs)
    candidates = {}  # isbn -> (possibly shrunk) similarity
    for dist, neighbor_idx in zip(distances.flatten(), indices.flatten()):
        if neighbor_idx in query_set:
            continue
        isbn = isbn_dec[neighbor_idx]
        sim = 1 - dist  # cosine distance -> similarity
        if shrinkage:
            # how many of this candidate's raters are also raters of the query work
            shared = pooled[item_user_sparse[neighbor_idx].indices].sum()
            sim *= min(shared, shrinkage) / shrinkage
        if isbn not in candidates or sim > candidates[isbn]:
            candidates[isbn] = sim

    # Filter to books with enough ratings, then take top-N
    valid_isbns = allowed_isbns(books_df, min_ratings, languages)
    candidates = {isbn: s for isbn, s in candidates.items() if isbn in valid_isbns}
    top = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:n_recs]

    result = pd.DataFrame(top, columns=['ISBN', 'Score'])
    result = result.merge(books_df[['ISBN', 'Title', 'Author', 'TotalRating', 'Subjects', 'Rel_Cum_Rating']], on='ISBN')
    result = result[result['Rel_Cum_Rating'] >= min_rcr]  # display-only: keep books with mean rating >= min_rcr
    result = result[['ISBN', 'Title', 'Score', 'Author', 'Subjects', 'TotalRating', 'Rel_Cum_Rating']]
    # Score = similarity
    return filter_same_work(result, title_query)

def content_based(title_query, n_recs=N_RECS, min_ratings=5, min_rcr=MIN_REL_CUM_RATING, languages=None):
    """
    Content-Based: TF-IDF cosine similarity on Title + Author + Subjects.
    
    Averages the TF-IDF vectors of all editions matching the query to form
    a single query profile, then finds the most similar books.
    """
    query_isbns = find_isbns_by_title(title_query, books_content)
    query_idxs = [content_isbn_to_idx[isbn] for isbn in query_isbns if isbn in content_isbn_to_idx]
    
    if not query_idxs:
        print(f"No books found matching '{title_query}'")
        return pd.DataFrame()
    
    print(f"Content query: {len(query_idxs)} editions")
    
    # Average TF-IDF vector across all query editions (convert from matrix to array)
    query_vec = np.asarray(tfidf_matrix[query_idxs].mean(axis=0))
    
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    scored = pd.Series(sims, index=content_isbn_list) #content_isbn_list = books_content['ISBN'].tolist()
    # removes the queried book's own editions from the results
    scored = scored.drop(labels=list(query_isbns), errors='ignore')
    
    valid_isbns = allowed_isbns(books_content, min_ratings, languages)
    scored = scored[scored.index.isin(valid_isbns)]
    
    top = scored.sort_values(ascending=False).head(n_recs * 3)
    
    result = pd.DataFrame({
        'ISBN': top.index,
        'Score': top.values
    }).reset_index(drop=True)
    result = result.merge(
        books_content[['ISBN', 'Title', 'Author', 'TotalRating', 'Subjects', 'Rel_Cum_Rating']], on='ISBN'
    )
    result = result[result['Rel_Cum_Rating'] >= min_rcr]  # display-only: keep books with mean rating >= min_rcr
    result = result[['ISBN', 'Title', 'Score', 'Author', 'Subjects', 'TotalRating', 'Rel_Cum_Rating']]
    result = filter_same_work(result, title_query)
    return result.head(n_recs)

def svd_recommend(title_query, books_df=books, ratings_df=ratings,
                  n_recs=N_RECS, fan_threshold=8, min_ratings=5, min_rcr=MIN_REL_CUM_RATING, languages=None):
    """
    SVD fan-based recommendation: reconstruct predicted ratings for fans of the query book,
    then recommend items with the highest average predicted rating.
    The ratings considered for fan_thresholdare as follows, 
    10: The Best
    9-8: Amazing (Outstanding, highly impressive)
    7-6: Good (Solid, reliable, above average)
    5-4: OK (Average, passable, needs minor improvements)
    3-2: Bad (Significantly flawed, problematic)
    1: Worst (Complete failure, unacceptable)
    """
    query_isbns = find_isbns_by_title(title_query, books_df)
    if not query_isbns:
        print(f"No books found matching '{title_query}'")
        return pd.DataFrame()

    # Find fans: users who rated any edition of the query book >= fan_threshold
    fan_ratings = ratings_df[ratings_df['ISBN'].isin(query_isbns)]
    fans = fan_ratings[fan_ratings['Rating'] >= fan_threshold]['UserID'].unique()
    fan_indices = [user_enc[uid] for uid in fans if uid in user_enc]

    if len(fan_indices) < 2:
        print(f"Not enough fans for '{title_query}' (found {len(fan_indices)})")
        return pd.DataFrame()

    print(f"SVD query: '{title_query}' → {len(fan_indices)} fans (rated >= {fan_threshold})")

    # Predicted rating for user u, item i: r_hat(u,i) = mean_u + (U*Sigma)_u @ Vt_i
    # Average across fans (linearity lets us average embeddings first):
    #   avg_pred_i = mean(mean_u for fans) + mean((U*Sigma)_u for fans) @ Vt_i
    avg_fan_mean = user_means[fan_indices].mean()
    avg_fan_embedding = U_sigma[fan_indices].mean(axis=0)  # shape: (k,)
    avg_pred = avg_fan_mean + avg_fan_embedding @ Vt  # shape: (n_items,)

    # Exclude query book editions
    query_idxs = {isbn_enc[isbn] for isbn in query_isbns if isbn in isbn_enc}
    for idx in query_idxs:
        avg_pred[idx] = -np.inf

    # Filter to books with enough ratings
    valid_isbns = allowed_isbns(books_df, min_ratings, languages)

    scored = {}
    for idx in np.argsort(-avg_pred):
        isbn = isbn_dec.get(idx)
        if isbn is None or isbn in query_isbns:
            continue
        if isbn not in valid_isbns:
            continue
        scored[isbn] = avg_pred[idx]
        if len(scored) >= n_recs:
            break

    result = pd.DataFrame({
        'ISBN': list(scored.keys()),
        'PredictedRating': list(scored.values())
    }).reset_index(drop=True)
    result = result.merge(books_df[['ISBN', 'Title', 'Author', 'TotalRating', 'Subjects', 'Rel_Cum_Rating']], on='ISBN')
    result = result[result['Rel_Cum_Rating'] >= min_rcr]  # display-only: keep books with mean rating >= min_rcr
    result = result[['ISBN', 'Title', 'PredictedRating', 'Author', 'Subjects', 'TotalRating', 'Rel_Cum_Rating']]
    return filter_same_work(result, title_query)

def user_based_cf(title_query, ratings_df=ratings, books_df=books,
                  n_recs=N_RECS, fan_threshold=8, min_ratings=10, K=18, min_rcr=MIN_REL_CUM_RATING, languages=None):
    """
    User-Based CF: per-fan KNN with KNNWithMeans prediction.

    1. Find fans (users who rated the queried title >= fan_threshold).
    2. Fit KNN (cosine on mean-centered data ≈ Pearson) on all users.
    3. For each fan, find K nearest neighbors and predict ratings
       for items those neighbors rated.
    4. Average raw predictions across fans .
    """
    exclude_isbns = find_isbns_by_title(title_query, books_df)
    if not exclude_isbns:
        print(f"No books found matching '{title_query}'")
        return pd.DataFrame()

    fan_ratings = ratings_df[ratings_df['ISBN'].isin(exclude_isbns)]
    fans = fan_ratings[fan_ratings['Rating'] >= fan_threshold]['UserID'].unique()
    fan_indices = [user_enc[uid] for uid in fans if uid in user_enc]

    if len(fan_indices) < 2:
        print("Not enough fans for similarity computation.")
        return pd.DataFrame()

    print(f"Fans of '{title_query}' (rated >= {fan_threshold}): {len(fan_indices)}")

    # Mean-center all users' ratings (cosine on centered data ≈ Pearson)
    u_sums = np.asarray(user_item_sparse.sum(axis=1)).flatten()
    u_counts = np.asarray((user_item_sparse > 0).sum(axis=1)).flatten()
    u_means = u_sums / u_counts #average rating per user

    r_nz, c_nz = user_item_sparse.nonzero()
    centered_vals = user_item_sparse.data.astype(np.float64) - u_means[r_nz]
    user_item_centered = csr_matrix(
        (centered_vals, (r_nz, c_nz)), shape=user_item_sparse.shape
    )

    # Fit KNN on all users, query for each fan individually
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=K + 1)
    knn.fit(user_item_centered)

    # Batch query: one row per fan
    fan_centered = user_item_centered[fan_indices]
    distances, indices = knn.kneighbors(fan_centered)  # (n_fans, K+1)

    # For each fan, predict ratings via their own neighbors
    fan_item_preds = {}  # isbn -> list of per-fan predictions

    for fan_i, fan_idx in enumerate(fan_indices):
        fan_mean = u_means[fan_idx]

        # Collect this fan's valid neighbors (skip self)
        neighbors = []
        for j in range(distances.shape[1]):
            n_idx = indices[fan_i, j]
            if n_idx == fan_idx:
                continue
            sim = 1 - distances[fan_i, j]
            if sim <= 0:
                continue
            neighbors.append((n_idx, sim))

        if not neighbors:
            continue

        # KNNWithMeans: pred(i) = mean_fan + Σ[sim * (r(v,i) - mean_v)] / Σ[sim]
        item_wdev = {}   # item_idx -> weighted deviation sum
        item_simsum = {} # item_idx -> similarity sum
        # calculation for all neighbors of each fan
        for n_idx, sim in neighbors:
            n_vec = np.asarray(user_item_sparse[n_idx].todense()).flatten()
            rated = np.where(n_vec > 0)[0]
            if len(rated) == 0:
                continue
            n_mean = n_vec[rated].mean()
            for item_idx in rated:
                dev = n_vec[item_idx] - n_mean
                item_wdev[item_idx] = item_wdev.get(item_idx, 0.0) + sim * dev
                item_simsum[item_idx] = item_simsum.get(item_idx, 0.0) + sim

        # Raw prediction
        for item_idx, wdev in item_wdev.items():
            if item_simsum[item_idx] <= 0:
                continue
            isbn = isbn_dec.get(item_idx)
            if isbn is None or isbn in exclude_isbns:
                continue
            pred = fan_mean + wdev / item_simsum[item_idx]
            if isbn not in fan_item_preds:
                fan_item_preds[isbn] = []
            fan_item_preds[isbn].append(pred) #predicted rating per item for a fan

    # Average predictions across fans
    scores = {isbn: np.mean(preds) for isbn, preds in fan_item_preds.items()}

    valid_isbns = allowed_isbns(books_df, min_ratings, languages)
    scored = pd.Series({isbn: s for isbn, s in scores.items() if isbn in valid_isbns})
    scored = scored.sort_values(ascending=False).head(n_recs)

    result = pd.DataFrame({
        'ISBN': scored.index,
        'CFScore': scored.values
    }).reset_index(drop=True)
    result = result.merge(books_df[['ISBN', 'Title', 'Author', 'TotalRating', 'Subjects', 'Rel_Cum_Rating']], on='ISBN')
    result = result[result['Rel_Cum_Rating'] >= min_rcr]  # display-only: keep books with mean rating >= min_rcr
    result = result[['ISBN', 'Title', 'CFScore', 'Author', 'Subjects', 'TotalRating', 'Rel_Cum_Rating']]
    return filter_same_work(result, title_query)

def lightgbm_recommend(title_query, books_df=books, n_recs=N_RECS,
                       fan_threshold=8, min_ratings=3, max_fans=100, min_rcr=MIN_REL_CUM_RATING, languages=None):
    """
    Title-based recommendation via the trained LightGBM, returning the top-N.

    Prototype-fan trick (for speed): instead of predicting for every fan and
    averaging (cost ~ n_fans x n_cand), we average the fans into ONE centroid in
    feature space and predict a single time. LightGBM's inputs are SVD-derived
    (which average linearly), so this mirrors svd_recommend's avg-embedding
    approach -- an approximation of the per-fan average that is ~n_fans times
    faster. `max_fans` therefore now affects only centroid quality, not runtime.
    """
    q = find_isbns_by_title(title_query, books_df)
    fans = ratings[(ratings['ISBN'].isin(q)) & (ratings['Rating'] >= fan_threshold)]['UserID'].unique()
    fan_idx = [user_enc[u] for u in fans if u in user_enc]
    if len(fan_idx) < 2:
        print(f"Not enough fans for '{title_query}'")
        return pd.DataFrame()
    if len(fan_idx) > max_fans:
        fan_idx = list(rng_lgb.choice(fan_idx, max_fans, replace=False))
    print(f"LightGBM '{title_query}': single predict for the centroid of {len(fan_idx)} fans")
    # candidate books
    cand = np.array([isbn_enc[i] for i in allowed_isbns(books_df, min_ratings, languages)
                     if i in isbn_enc])
    # Prototype fan: average the fans' features into one centroid, then predict ONCE.
    cnt_avg  = lgb_u_cnt[fan_idx].mean()                       # avg rating count
    mean_avg = user_means[fan_idx].mean()                      # avg mean rating
    emb_avg  = U_sigma[fan_idx].mean(axis=0)                   # centroid SVD embedding (k,)
    svd_pred = mean_avg + emb_avg @ Vt[:, cand]                # SVD predicted rating per candidate
    base = np.column_stack([np.full(len(cand), cnt_avg),
                            np.full(len(cand), mean_avg), svd_pred])
    feats = np.hstack([base, np.tile(emb_avg, (len(cand), 1)), Vt[:, cand].T])
    acc = lgb_model.predict(feats)                             # one batched predict over all candidates

    qset = set(q); rows = []
    for o in np.argsort(-acc):
        isbn = isbn_dec[cand[o]]
        if isbn in qset: #drop books from the query
            continue
        rows.append((isbn, acc[o]))
        if len(rows) >= n_recs * 3:
            break
    res = pd.DataFrame(rows, columns=['ISBN', 'PredictedRating']).merge(
        books_df[['ISBN', 'Title', 'Author', 'TotalRating', 'Subjects', 'Rel_Cum_Rating']], on='ISBN')
    res = res[res['Rel_Cum_Rating'] >= min_rcr]  # display-only: keep books with mean rating >= min_rcr
    res = res[['ISBN', 'Title', 'PredictedRating', 'Author', 'Subjects', 'TotalRating', 'Rel_Cum_Rating']]
    return filter_same_work(res, title_query).head(n_recs)


# ============================ hybrid ============================

def hybrid_recommend(title_query, n_recs=N_RECS, min_rcr=MIN_REL_CUM_RATING, n_per_method=20, k=60, max_per_family=2,
                    # w_item=0.833, w_content=0.016, w_svd=0.044, w_user=0.097, w_lgbm=0.011, languages=None): # metric based
                    w_item=0.1, w_content=0.33, w_svd=0.22, w_user=0.10, w_lgbm=0.25, languages=None): # human based
    """
    Hybrid via Reciprocal Rank Fusion (RRF) over five base methods.
        RRF(i) = Σ_method  w_method * 1 / (k + rank_method(i))
    RRF depends only on rank, so the methods' differing score scales don't matter.
    """
    print(f"Running hybrid (RRF) for '{title_query}' (weights: svd={w_svd}, user={w_user}, "
          f"item={w_item}, content={w_content}, lgbm={w_lgbm}, k={k})")
    print("-" * 100)
    _scope = "all" if languages == "all" else (set(DEFAULT_LANGUAGES) if languages is None else languages)
    #print(f"  Language scope: {_scope}")

    method_lists = {
        'Item-CF':  (item_based_cf(title_query, n_recs=n_per_method, min_rcr=min_rcr, languages=languages),   w_item),
        'Content':  (content_based(title_query, n_recs=n_per_method, min_rcr=min_rcr, languages=languages),   w_content),
        'SVD':      (svd_recommend(title_query, n_recs=n_per_method, min_rcr=min_rcr, languages=languages),   w_svd),
        'User-CF':  (user_based_cf(title_query, n_recs=n_per_method, min_rcr=min_rcr, languages=languages),   w_user),
        'LightGBM': (lightgbm_recommend(title_query, n_recs=n_per_method, min_rcr=min_rcr, languages=languages), w_lgbm),
    }

    RRF = {}  # isbn -> accumulated RRF score
    hits = {} # isbn -> list of contributing methods
    for name, (df, weight) in method_lists.items():
        if df.empty:
            continue
        for rank, isbn in enumerate(df['ISBN'].tolist()):
            RRF[isbn] = RRF.get(isbn, 0.0) + weight * (1.0 / (k + rank))
            hits.setdefault(isbn, []).append(name) # appends method's name as a list for each ISBN

    if not RRF: # True when RRF is empty
        print(f"No candidates produced for '{title_query}'") 
        return pd.DataFrame()

    result = pd.DataFrame({
        'ISBN': list(RRF.keys()),
        'HybridScore': list(RRF.values()),
        'Methods': [', '.join(hits[isbn]) for isbn in RRF],
        'nMethods': [len(hits[isbn]) for isbn in RRF],
    })
    result = result.sort_values('HybridScore', ascending=False).head(n_recs * 3).reset_index(drop=True)
    result = result.merge(books[['ISBN', 'Title', 'Author', 'TotalRating', 'Subjects']], on='ISBN')
    result = result[['ISBN', 'Title', 'HybridScore', 'nMethods', 'Methods', 'Author', 'TotalRating', 'Subjects']]
    result = filter_same_work(result, title_query)
    result = cap_per_family(result, max_per_family=max_per_family)  # diversify: cap each series/family

    hs = result['HybridScore']
    if len(hs) > 1 and hs.max() != hs.min():
        result['HybridScore'] = (hs - hs.min()) / (hs.max() - hs.min())
    return result.head(n_recs).reset_index(drop=True)



# ============================ convenience API ============================
def load_recommender(n_recs=N_RECS, **hybrid_kwargs):
    """Return a recommend(title) -> DataFrame[ISBN, Title] closure (silent)."""
    def recommend(title, n=n_recs):
        with contextlib.redirect_stdout(io.StringIO()):   # silence per-method diagnostics
            res = hybrid_recommend(title, n_recs=n, **hybrid_kwargs)
        if res is None or res.empty:
            return pd.DataFrame(columns=["ISBN", "Title"])
        return res[["ISBN", "Title"]].reset_index(drop=True)
    return recommend
