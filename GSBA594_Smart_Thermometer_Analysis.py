# %% [markdown]
# # GSBA 594 — November 15 In-Class Assignment
# ## Smart grilling thermometers: MEATER Plus vs. TempPro TempSpike TP960
# 
# **Business question:** For two competing app-connected cooking thermometers, what experiences distinguish satisfaction, frustration, and perceived value when their aggregate star ratings are similar?
# 
# Both products are consumer-tech/smart devices: a wire-free probe measures **internal food temperature and ambient cooking temperature**, relays readings through a nearby booster/charger to a phone app, and provides cooking alerts. They support more consistent cooking by informing the cook; **neither automatically regulates a grill's burners or airflow**. They are comparable single-probe products in the same entry-level smart-thermometer price tier. This replaces the previous conventional-griddle comparison.
# 
# | Product | Amazon product page | Comparable price evidence |
# |---|---|---|
# | MEATER Plus | [B07H8WTFHW](https://www.amazon.com/dp/B07H8WTFHW) | Manufacturer's US sale page: $59.99 |
# | TempPro TempSpike TP960 (formerly ThermoPro) | [B0B48NL6L8](https://www.amazon.com/dp/B0B48NL6L8) | Manufacturer's US homepage/collection: $64.99 |
# 
# Price sources checked November 15, 2026: [MEATER US sale page](https://www.meater.com/sale), [TempPro homepage](https://temppro.com/), [TempPro smart-thermometer collection](https://temppro.com/collections/smart-meat-thermometer?page=2). These are manufacturer advertised prices used to establish price tier, **not a claim of identical live Amazon checkout prices**. Manufacturer pages can disagree during promotions; the TP960 detail page has also displayed $79.99. Prices, coupons, and sellers change. Both selected products are sold on Amazon, and their category and single-probe use case match.
# 
# Feature sources: [MEATER Plus official product information](https://store-us.meater.com/products/meater-plus), [MEATER ambient-sensor documentation](https://support.meater.com/hc/en-us/articles/37182246162331-Ambient-Temperature-Sensor), [TempPro TP960 official product information](https://temppro.com/products/tp960-tempspike-bluetooth-meat-thermometer). Manufacturer specifications describe features; this project does not independently test sensor accuracy or wireless range.

# %% [markdown]
# ### Setup and reproducibility
# Keep `amazon_smart_thermometer_reviews.csv` beside this notebook and run all cells in order. The delivered notebook already contains the actual outputs. If packages are missing, run `%pip install pandas numpy matplotlib scipy scikit-learn wordcloud` in a separate cell and restart the kernel. No NLTK download, account login, live review connection, or threadpoolctl patch is required to run the analysis.

# %%
from pathlib import Path
import re
import html
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from wordcloud import WordCloud
from IPython.display import display, Markdown

pd.set_option('display.max_colwidth', 100)
plt.rcParams.update({'figure.figsize': (9, 5), 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 11})
RESULTS = Path('analysis_results')
RESULTS.mkdir(exist_ok=True)
df = pd.read_csv('amazon_smart_thermometer_reviews.csv', encoding='utf-8-sig')
df['review_date'] = pd.to_datetime(df['review_date'], errors='raise')
products = sorted(df['product_name'].unique())
colors = ['#167d9a', '#e38b32']
display(df.head(3))

# %% [markdown]
# ## Part A — Exploratory Data Analysis & Sampling Justification
# 
# ### Public Amazon-review collection
# The observations are **Amazon customer reviews republished in Amazon-owned Woot's “Amazon Ratings and Reviews” feeds**, keyed to the selected Amazon ASINs. They are not Woot customer comments, manufacturer testimonials, AI review summaries, or professional reviews. Direct Amazon programmatic access returned HTTP 503 during the earlier collection attempt; the public Woot feed provides the required text and structured metadata without inventing missing records.
# 
# **Predefined, content-independent collection rule:** For each product and each star level (1, 2, 3, 4, 5), retrieve the first five ten-review batches under both **Top** and **Most Recent**, with all reviewers included. Follow the feed's next-page cursor; numeric page parameters alone can repeat results. This gives up to 100 retrieval slots per rating per product (1,000 total slots). Remove overlap within and across sorts using a stable fingerprint of ASIN, author label, date, title, and full body. Do not select reviews based on the story they tell. Exclude empty bodies and dates after the collection date, November 15, 2026.
# 
# Recent sorting captures current experience; top sorting adds older, visible reviews. This deliberately oversamples lower-star reviews and helpful older reviews. It provides variation for an MBA-level lexical comparison, **not representative estimates of Amazon satisfaction**. It is a reproducible convenience/stratified sample, not a random sample or a complete census. Collection was automated extraction from public pages, disclosed because the PDF describes manually extracted data. The CSV has one row per distinct review fingerprint; source IDs are null in the feed, so exact Amazon review-ID deduplication is unavailable.
# 
# **Source limitations:** Woot does not expose per-review purchased-variant ASINs or original Amazon permalinks here. We know the listing ASIN, but cannot guarantee every review concerns the same color, bundle, generation, or hardware revision. Product names therefore identify the review listing; pooled variants and brand rebranding remain potential confounders. The evidence file retains the original review objects and retrieval URLs. Interpretations concern this review sample, not a laboratory comparison.

# %%
required = ['review_id', 'product_name', 'brand', 'star_rating', 'review_title',
            'review_text', 'review_date', 'verified_purchase', 'helpful_votes']
assert set(required).issubset(df.columns)
assert df['review_id'].is_unique
assert df[required].notna().all().all()
assert df['star_rating'].isin([1, 2, 3, 4, 5]).all()
assert df['verified_purchase'].isin([0, 1]).all()
assert (df['helpful_votes'] >= 0).all()
assert df['review_text'].str.strip().ne('').all()
print(f'Total unique reviews collected: {len(df):,}')
print('Duplicate body text within product (inspect if nonzero):',
      df.duplicated(['product_name', 'review_text']).sum())
display(df[required].isna().sum().rename('missing_values').to_frame())
summary = df.groupby('product_name').agg(
    total_reviews=('review_id', 'size'), mean_rating=('star_rating', 'mean'),
    median_rating=('star_rating', 'median'), verified_share=('verified_purchase', 'mean'),
    first_review=('review_date', 'min'), last_review=('review_date', 'max'))
summary['verified_pct'] = 100 * summary.pop('verified_share')
display(summary.round(2))
summary.to_csv(RESULTS / 'part_a_summary.csv')

# %%
rating_counts = pd.crosstab(df['product_name'], df['star_rating']).reindex(columns=range(1, 6), fill_value=0)
rating_pct = rating_counts.div(rating_counts.sum(axis=1), axis=0) * 100
print('Review counts by star rating:')
display(rating_counts)
print('Percent within each collected product sample:')
display(rating_pct.round(1))
rating_pct.to_csv(RESULTS / 'rating_percentages.csv')
ax = rating_pct.T.plot.bar(color=colors)
ax.set(xlabel='Star rating', ylabel='Percent of collected reviews', title='Sample rating distributions')
ax.tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(RESULTS / 'rating_distribution.png', dpi=160)
plt.show()

coverage = pd.crosstab([df['product_name'], df['review_date'].dt.year], df['star_rating']).reindex(columns=range(1, 6), fill_value=0)
print('Year × star coverage (zeros are real coverage gaps):')
display(coverage)
coverage.to_csv(RESULTS / 'date_rating_coverage.csv')
for product in products:
    absent = rating_counts.columns[rating_counts.loc[product].eq(0)].tolist()
    print(product, 'missing star categories:', absent or 'none')

# %%
helpful_summary = df.groupby('product_name')['helpful_votes'].describe(percentiles=[.25, .5, .75, .9])
helpful_summary['zero_votes_pct'] = df.groupby('product_name')['helpful_votes'].apply(lambda s: 100 * s.dropna().eq(0).mean())
display(helpful_summary.round(2))
helpful_summary.to_csv(RESULTS / 'helpful_vote_summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for product, color in zip(products, colors):
    values = df.loc[df['product_name'].eq(product), 'helpful_votes'].dropna()
    axes[0].hist(values, bins=15, alpha=.6, label=product, color=color)
axes[0].set(xlabel='Helpful votes', ylabel='Reviews', title='Helpful votes: raw distribution')
axes[0].legend()
axes[1].boxplot([np.log1p(df.loc[df['product_name'].eq(p), 'helpful_votes'].dropna()) for p in products], tick_labels=products)
axes[1].set(ylabel='log(1 + helpful votes)', title='Helpful votes: compressed scale')
plt.tight_layout()
plt.savefig(RESULTS / 'helpful_votes.png', dpi=160)
plt.show()

# %% [markdown]
# **Interpretation:** The sample means, medians, and rating percentages describe the deliberately collected written reviews. They should not reproduce the listings' overall ratings: low-star reviews are intentionally well represented, duplicates across sorts were removed, and ratings without text are absent. Global star ratings are not raw satisfaction benchmarks for this sample. Broad date coverage is reported in the year × rating table; it does not guarantee even coverage in every period.
# 
# Helpful votes are right-skewed and influenced by review age and the Top sort. Use medians, quantiles, and zero-vote shares alongside means. Helpful counts and verified badges are reported directly from the feed; no helpful votes means zero. A verified-purchase label does not eliminate selection bias. Vine reviews, when present, are flagged separately; free-product reviews can have different incentives. All text is retained in the evidence file; source text claims are not independently verified.

# %%
print('Country coverage:')
display(pd.crosstab(df['product_name'], df['review_country']))
print('Vine/free-product reviews:')
display(df.groupby('product_name')['vine_review'].agg(['sum', 'mean']).rename(columns={'sum':'vine_reviews','mean':'vine_share'}))
print('Unique reviews by rating and year:')
display(df.groupby('product_name').agg(years=('review_date', lambda s: s.dt.year.nunique()),
                                      total_body_words=('review_text', lambda s: s.str.split().str.len().sum())))

# %% [markdown]
# ### Why this sample is adequate—and where it is weak
# Every product has observations at each of the five star levels and across multiple years, allowing basic within-product correlation and contrasting vocabulary across the full satisfaction range. Hundreds of unique documents support recurring term/bigram summaries more credibly than a few selected anecdotes. The final counts and text volume appear above. This is adequate for **exploratory** TF-IDF analysis; it does not establish statistical power for small associations, stable latent topics, causal drivers, or population-level generalization. Unequal product histories, sort visibility, app updates, pooled variants, and reviewer self-selection remain limitations. No reviews are fabricated to fill a star/date cell.

# %% [markdown]
# ## Part B — Domain-Tailored Text Preprocessing
# 
# Analyze review **body only**, preserving title and raw body in the source CSV. Avoid double-weighting an idea when a title repeats it. Lowercase, decode HTML, normalize Unicode, remove URLs, expand common negations, and normalize Wi-Fi/Bluetooth spelling. Remove manufacturer/model names and generic product labels so a brand name cannot trivially distinguish its own product. Keep experience words such as **app, connection, ambient, internal, accuracy, battery, charging, range, alerts, cooking, and temperature**. Retain negations (not/no/never/without) because “not accurate” differs from “accurate.” Remove punctuation and standalone numbers; temperature values are not directly compared as engineering measurements. No stemming is used so the tables are easy to read.
# 
# Removing stopwords before forming bigrams can join words that were not adjacent in the original sentence. Treat these as adjacent **retained-token pairs**, and inspect examples before interpreting them as literal phrases. Part D counts raw-body words **before** cleaning or removing stopwords.

# %%
keep_words = {'not', 'no', 'nor', 'never', 'without'}
domain_stopwords = {'meater', 'thermopro', 'temppro', 'tempspike', 'spike', 'temp', 'tp',
                    'plus', 'thermometer', 'thermometers', 'probe', 'probes', 'product',
                    'products', 'amazon', 'bought', 'purchase', 'purchased', 'got',
                    'does', 'just', 'really', 'thing', 'things'}
stop_words = (set(ENGLISH_STOP_WORDS) - keep_words) | domain_stopwords

def clean_review(text):
    text = html.unescape(unicodedata.normalize('NFKC', str(text))).lower()
    text = text.replace('’', "'")
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r"\b(can't|cannot)\b", 'can not', text)
    text = re.sub(r"\bwon't\b", 'will not', text)
    text = re.sub(r"n't\b", ' not', text)
    text = re.sub(r'wi[\s-]*fi', 'wifi', text)
    text = re.sub(r'blue[\s-]*tooth', 'bluetooth', text)
    text = re.sub(r'°\s*f\b', ' fahrenheit ', text)
    text = re.sub(r'°\s*c\b', ' celsius ', text)
    tokens = re.findall(r'[a-z]+', text)
    return ' '.join(t for t in tokens if len(t) > 1 and t not in stop_words)

df['clean_text'] = df['review_text'].map(clean_review)
df['review_word_count'] = df['review_text'].map(
    lambda text: len(re.findall(r"\b\w+(?:['’]\w+)*\b", str(text))))
display(df[['product_name', 'review_text', 'clean_text', 'review_word_count']].head(5))
print('Reviews with no lexical tokens after cleaning:', df['clean_text'].eq('').sum())
df.to_csv(RESULTS / 'reviews_with_nlp_features.csv', index=False)

# %% [markdown]
# ## Part C — Comparative Lexical & Topic Analysis
# 
# Fit **one shared TF-IDF vocabulary**, using unigrams/bigrams appearing in at least three documents. Each review's vector has unit length, so long reviews do not dominate only because of their length. For each product, first compute mean TF-IDF within each star level, then average the five means with equal weights. This **rating-standardized** score puts both products on the same star-rating mix, reducing confounding from small differences in the deduplicated samples' rating composition.
# 
# Subtract the competing product's rating-standardized mean from the product's own mean. Its top 15 positive differences are the most distinctive terms/bigrams under this definition. Report both products' document counts to distinguish recurring language from rare words. A positive contrast means relatively more prominent, not exclusive, causal, or statistically significant. WordClouds use the same positive contrasts. TF-IDF is the selected method allowed by the PDF; a separate latent-topic model is not required.

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=1.0,
                             token_pattern=r'(?u)\b[a-z][a-z]+\b', lowercase=False)
X = vectorizer.fit_transform(df['clean_text'])
terms = vectorizer.get_feature_names_out()
means = {}
document_counts = {}
for product in products:
    mask = df['product_name'].eq(product).to_numpy()
    # Give each star level equal weight in the product's vocabulary profile.
    star_means = []
    for star in range(1, 6):
        star_mask = mask & df['star_rating'].eq(star).to_numpy()
        star_means.append(np.asarray(X[star_mask].mean(axis=0)).ravel())
    means[product] = np.mean(star_means, axis=0)
    document_counts[product] = np.asarray((X[mask] > 0).sum(axis=0)).ravel()

top_tables = {}
for product, color in zip(products, colors):
    competitor = next(p for p in products if p != product)
    difference = means[product] - means[competitor]
    table = pd.DataFrame({'term_or_bigram': terms, 'mean_tfidf': means[product],
                          'other_mean_tfidf': means[competitor], 'difference': difference,
                          'reviews_with_term': document_counts[product],
                          'other_reviews_with_term': document_counts[competitor]})
    table = table[table['difference'] > 0].sort_values(['difference', 'term_or_bigram'], ascending=[False, True])
    top_tables[product] = table.head(15)
    display(Markdown(f'### {product}: top 15 distinctive terms or bigrams'))
    display(top_tables[product].round(4).reset_index(drop=True))
    table.head(15).to_csv(RESULTS / f'{product}_top15_terms.csv', index=False)
    ax = table.head(15).iloc[::-1].plot.barh(x='term_or_bigram', y='difference', color=color, legend=False)
    ax.set(xlabel='Rating-standardized TF-IDF advantage', ylabel='', title=product)
    plt.tight_layout()
    plt.savefig(RESULTS / f'{product}_distinctive_terms.png', dpi=160)
    plt.show()
    weights = dict(zip(table['term_or_bigram'], table['difference']))
    cloud = WordCloud(width=1100, height=550, background_color='white',
                      colormap='viridis', max_words=60, random_state=594,
                      collocations=False).generate_from_frequencies(weights)
    cloud.to_file(str(RESULTS / f'{product}_wordcloud.png'))
    plt.figure(figsize=(11, 5.5))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{product}: distinctive review language')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Read distinctive words in context
# The following cell displays three reviews with the strongest distinctive vocabulary per product, alongside ratings and provenance URLs. These examples explain what terms mean; they do not replace the full sample. Read positive and negative contexts before assigning a theme. Complaints about accuracy, connectivity, or food outcomes are customer reports, not experimentally verified failures.

# %%
for product in products:
    competitor = next(p for p in products if p != product)
    weights = np.maximum(means[product] - means[competitor], 0)
    scores = np.asarray(X @ weights).ravel()
    examples = df.loc[df['product_name'].eq(product)].copy()
    examples['lexical_score'] = scores[df['product_name'].eq(product).to_numpy()]
    display(Markdown(f'**{product}: contextual examples**'))
    display(examples.nlargest(3, 'lexical_score')[['star_rating', 'review_title', 'review_text', 'source_url']])

# %% [markdown]
# ## Part D — Integrating Unstructured Features With Structured Data
# 
# **Question:** Is star rating correlated with review word count?
# 
# Use **Spearman correlation** as the primary test: stars are ordinal and review lengths are skewed. Report Pearson as a secondary linear summary. Test the null of zero association versus a two-sided alternative, pooled and within each product. The within-product results are more interpretable because they reduce product-mix confounding. Jitter is added only to plotted star positions, never to test inputs. Raw word count is an NLP-derived feature, not a sentiment score.
# 
# This deliberately stratified, visibility-selected sample is not independent random sampling from buyers; treat p-values as exploratory model-based summaries. Multiple tests are unadjusted. Association does not establish that dissatisfaction causes longer reviews, and the data cannot measure actual cooking-temperature consistency.

# %%
correlation_rows = []
groups = [('Pooled', df)] + [(p, df[df['product_name'].eq(p)]) for p in products]
for label, group in groups:
    x, y = group['star_rating'], group['review_word_count']
    if len(group) >= 3 and x.nunique() > 1 and y.nunique() > 1:
        rho, spearman_p = stats.spearmanr(x, y)
        r, pearson_p = stats.pearsonr(x, y)
    else:
        rho = spearman_p = r = pearson_p = np.nan
    correlation_rows.append({'sample': label, 'n': len(group), 'spearman_rho': rho,
                             'spearman_p': spearman_p, 'pearson_r': r, 'pearson_p': pearson_p})
correlations = pd.DataFrame(correlation_rows)
display(correlations.round(4))
correlations.to_csv(RESULTS / 'part_d_correlations.csv', index=False)
length_summary = df.groupby(['product_name', 'star_rating'])['review_word_count'].agg(['count', 'mean', 'median', 'min', 'max'])
display(length_summary.round(1))
length_summary.to_csv(RESULTS / 'word_count_by_rating.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
rng = np.random.default_rng(594)
for product, color in zip(products, colors):
    group = df[df['product_name'].eq(product)]
    axes[0].scatter(group['star_rating'] + rng.uniform(-.12, .12, len(group)),
                    group['review_word_count'], alpha=.55, color=color, label=product)
    medians = group.groupby('star_rating')['review_word_count'].median().reindex(range(1,6))
    axes[1].plot(medians.index, medians, marker='o', color=color, label=product)
axes[0].set(xlabel='Star rating (jittered for visibility)', ylabel='Raw body word count', title='Review length and rating')
axes[1].set(xlabel='Star rating', ylabel='Median raw body word count', title='Median length within each star group')
for ax in axes:
    ax.set_xticks(range(1, 6))
    ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / 'rating_vs_word_count.png', dpi=160)
plt.show()
for row in correlation_rows:
    direction = 'negative' if row['spearman_rho'] < 0 else 'positive'
    evidence = 'below' if row['spearman_p'] < .05 else 'not below'
    print(f"{row['sample']}: Spearman rho={row['spearman_rho']:.3f}, "
          f"p={row['spearman_p']:.4g}, n={row['n']}; {direction} association. "
          f"The exploratory p-value is {evidence} 0.05.")

# %% [markdown]
# ### Sensitivity check: reviews seen in the Most Recent stream
# Top-ranked reviews can be longer and more helpful. Recalculate Spearman using only reviews encountered in a Most Recent batch. This subset is still star-stratified and still not representative, but helps show whether the sorting choice materially changes the association.

# %%
recent_rows = []
recent = df[df['retrieval_groups'].str.contains('recent')]
for product in products:
    group = recent[recent['product_name'].eq(product)]
    rho, p = stats.spearmanr(group['star_rating'], group['review_word_count'])
    recent_rows.append({'product': product, 'n': len(group), 'spearman_rho': rho, 'p_value': p})
recent_correlations = pd.DataFrame(recent_rows)
display(recent_correlations.round(4))
recent_correlations.to_csv(RESULTS / 'recent_stream_correlations.csv', index=False)

# %% [markdown]
# ## Observed findings and answers
# 
# **Part A:** There are 715 unique reviews: 375 MEATER Plus (December 19, 2018–November 13, 2026) and 340 TempPro TempSpike TP960 (September 4, 2022–November 12, 2026). Both have every star level. Sample mean ratings are 2.87 and 2.83, respectively; both medians are 3. These low means reflect deliberate low-star oversampling, not the listings' overall satisfaction. Verified-purchase shares are 85.9% and 92.4%. All retrieved reviews are labeled United States; no Vine reviews were flagged. There were 950 successful retrieval slots before overlap removal. The final 50 planned slots could not be completed: the feed returned HTTP 403 when requesting TempSpike's fifth Top 5-star batch, and later uncached batches were not requested. The collection log records this limitation.
# 
# **Part B:** The text-cleaning approach retains technical experience language and negations while removing brand/model identifiers, common filler, and special characters. Word count remains based on the original review body.
# 
# **Part C:** MEATER's distinctive vocabulary emphasizes connection, phone/app use, Bluetooth, and cooking outcomes. The word “connection” occurs in 89 MEATER reviews and 33 TempSpike reviews. TempSpike's distinctive vocabulary emphasizes charge/charged, booster, battery, hold charge, replacement, and elapsed use (months/year). “Hold charge” appears in 26 TempSpike reviews and three MEATER reviews. These counts support candidate themes; the ranking uses equal-star-weight TF-IDF. Connectivity and cooking outcomes warrant contextual review for MEATER; charging endurance, pairing, and replacement/support experiences warrant contextual review for TempSpike. Recommendations are to investigate these friction points and clarify setup, range, charging, and ambient-temperature expectations. This does not establish that one sensor is more accurate or that a particular design change would cause higher satisfaction.
# 
# **Part D:** Pooled Spearman rho = -0.113 (p = 0.00256, n = 715), a weak negative association. Within MEATER, rho = -0.262 (p = 2.70 × 10^-7, n = 375): lower-star reviews tend to be longer in this collected frame. Within TempSpike, rho = +0.098 (p = 0.0716, n = 340), which does not provide clear evidence at the exploratory 0.05 threshold. Pearson is weaker (MEATER r = -0.111; TempSpike r = +0.090). The Most Recent-only subset preserves MEATER's negative association (rho = -0.301), while TempSpike is near zero (rho = -0.028). That sensitivity cautions against a general statement that negative reviews are always longer.
# 
# **Practical limit:** The estimates summarize accessible, rating-stratified Amazon reviews with Top/Recent visibility bias. They are not random-sample population estimates or measurements of actual grill temperature control. Review dates, model revisions, length, and ranking may affect the observed contrasts.
# 
# 
# **Submission:** Upload this output-filled notebook and `amazon_smart_thermometer_reviews.csv`. The README documents product choice, price evidence, sampling and access limits. Source records and batch URLs are in `review_evidence.json` and `collection_log.csv`; results and WordClouds are exported to `analysis_results/`.
