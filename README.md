# Smart-Thermometer-Reviews
# GSBA 594 — Smart Thermometer Submission

This package replaces the conventional-griddle assignment with **two competing consumer-tech/smart devices** for grilling. Both use an app-connected wireless probe to measure internal meat temperature and surrounding cooking temperature. They provide readings and alerts; neither automatically controls a grill's burners or airflow.

## Products and comparable price tier

- **MEATER Plus**, one probe: [Amazon B07H8WTFHW](https://www.amazon.com/dp/B07H8WTFHW).
- **TempPro TempSpike TP960**, one probe, also sold under the ThermoPro name: [Amazon B0B48NL6L8](https://www.amazon.com/dp/B0B48NL6L8).

Manufacturer US advertised prices checked November 15, 2026 were **$59.99** for MEATER Plus ([sale page](https://www.meater.com/sale)) and **$64.99** for TempSpike TP960 ([homepage](https://temppro.com/), [collection](https://temppro.com/collections/smart-meat-thermometer?page=2)). These establish a comparable entry-level price tier, not guaranteed live Amazon checkout prices. Coupons and seller discounts vary. TempPro's detail page has also displayed $79.99, so price evidence is promotional and page-dependent; even that is the same broad $60–$80 tier. This is a matched single-probe use-case comparison rather than a multi-probe kit versus a single unit.

Official feature sources: [MEATER Plus](https://store-us.meater.com/products/meater-plus), [MEATER ambient sensors](https://support.meater.com/hc/en-us/articles/37182246162331-Ambient-Temperature-Sensor), [TempPro TP960](https://temppro.com/products/tp960-tempspike-bluetooth-meat-thermometer). The analysis does not test engineering claims or establish food-safety guidance.

## Real Amazon reviews and sampling

Collected November 15, 2026 from Amazon-owned Woot's public **Amazon Ratings and Reviews** feeds for those ASINs. These are republished Amazon customer reviews, not Woot discussion comments or manufacturer testimonials. Feed pages: [MEATER](https://home.woot.com/review/B07H8WTFHW?itemid=TP17469A&offerid=af42b853-07fa-4082-81b8-bcbc78e2b37d), [TempSpike](https://home.woot.com/review/B0B48NL6L8?itemid=WT525166A&offerid=ee343c42-c983-4aa9-a3a0-8a1930033279). No review text, ratings, dates, badges, or helpful votes were fabricated. Direct programmatic Amazon access had returned HTTP 503. Public feed extraction was automated, disclosed because the PDF describes manual extraction.

The rule was fixed before interpretation: for every product × star level, take five ten-review batches in **Top** order and five in **Most Recent** order, all reviewers, using the public next-page cursor. Deduplicate overlap using ASIN + author label + displayed date + title + full body. Every retrieved valid review was retained, without selecting by narrative. Numeric pagination can repeat records; only the cursor-based expansion is used in the final collection.

The feed blocked the fifth Top 5-star TempSpike batch with HTTP 403. Further uncached batches were not requested after that barrier. This leaves 950 successful retrieval slots out of 1,000 planned, yielding **715 unique review fingerprints** after overlap removal. All are labeled United States. No Vine reviews were flagged.

| Product | 1-star | 2-star | 3-star | 4-star | 5-star | Total | Dates |
|---|---:|---:|---:|---:|---:|---:|---|
| MEATER Plus | 88 | 74 | 76 | 71 | 66 | 375 | 2018-12-19–2026-11-13 |
| TempSpike TP960 | 75 | 73 | 71 | 78 | 43 | 340 | 2022-09-04–2026-11-12 |

**All five ratings are present for both products.** The notebook also reports year × rating coverage. Hundreds of documents and broad rating/date coverage support exploratory TF-IDF and descriptive correlations. This is a stratified convenience sample with low-star and helpful-review oversampling, not a random sample of buyers. Mean/median ratings and distribution percentages describe this sample, not Amazon-wide satisfaction. Top-ranked reviews may be longer, older, and more helpful. Unadjusted p-values are exploratory and cannot establish causation or population representativeness.

The feed identifies listing ASINs but omits per-review purchased-variant ASINs and original Amazon review IDs. Pooled colors, bundles, generations, and hardware revisions cannot be ruled out; review-level permalinks cannot be reconstructed honestly. `source_url` points to the exact public feed request. The evidence JSON preserves raw review objects and retrieval URLs. IDs are stable hashes, not claimed original Amazon review IDs. Displayed date labels are used because the feed's numeric date field can be a 1970 placeholder. HTML entities are decoded; customer wording is preserved. Helpful counts and verified badges come directly from feed fields (no helpful votes = 0). No required values are missing.

## Parts A–D and results

- **A:** Counts, mean/median rating, rating percentages, verified share, helpful-vote distributions, date/rating coverage, and sampling limitations.
- **B:** Domain-tailored preprocessing with negation and technical terms preserved; raw-body word count kept separate.
- **C:** Shared unigram/bigram TF-IDF, top 15 distinctive terms per product, and two WordClouds. Each star level receives equal weight to reduce rating-mix confounding.
- **D:** Spearman and Pearson correlations, plots, and a Most Recent-stream sensitivity check.

MEATER's distinctive language emphasizes connectivity, app/phone use, and cooking outcomes. TempSpike emphasizes charging, booster, battery endurance, and replacements. Sample ratings average 2.87 and 2.83 (both median 3), reflecting sampling design. MEATER shows a modest negative star–length association (Spearman -0.262); TempSpike shows no clear association at the exploratory 0.05 threshold (+0.098, p=0.0716). The pooled association is weakly negative (-0.113). Detailed interpretations and limitations are in the notebook.

## Use and submit

Open `GSBA594_Smart_Thermometer_Analysis.ipynb` with the CSV beside it, install `requirements.txt` if necessary, and Run All. The notebook already contains outputs; all ten code cells were executed successfully in order and the file validated. HTML is included for reading without Jupyter; the `.py` provides the same code and explanatory comments. No computer control or Amazon login is needed to run the analysis. Submit the notebook and original CSV, plus these notes if allowed. `analysis_results/` contains standalone tables and charts; `collection_log.csv` and `review_evidence.json` provide provenance.
