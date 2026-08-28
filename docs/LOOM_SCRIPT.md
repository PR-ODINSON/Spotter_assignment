# Loom Video Script (2–3 minutes)

Read naturally. Timestamps are approximate guides.

---

**[0:00–0:20] Problem and dataset**

Hi, I'm walking through my freight rate prediction project for the Spotter ML assessment.

The goal is to predict the posted freight rate in dollars for each load, using shipment attributes like distance, equipment type, weight, and market signals.

I had forty-eight thousand labeled loads from January through October twenty twenty-five for development, and twelve thousand unlabeled loads from November and December for the final inference holdout.

The target is `posted_rate`. Each load has a unique `load_id`, which I never used as a model feature.

---

**[0:20–0:45] EDA findings**

In exploratory analysis, distance was the dominant predictor — Pearson correlation about zero point nine one with the target.

The target distribution is right-skewed and heavy-tailed: median around two thousand dollars, but long-haul loads can reach much higher legitimate rates.

Equipment mattered too — Reefer, Flatbed, and Dry Van have different typical rate levels.

Weight and quote signal added useful signal beyond distance alone.

---

**[0:45–1:10] Data quality**

I handled missing weight — about three hundred training rows — and negative weight values, which I treated as invalid and converted to missing.

I imputed weight by equipment median with a global fallback, always fit on the training fold only.

Market index had missing values and temporal drift, but after testing I excluded the raw market index from the final model because it hurt validation performance.

Validation also contains eight unseen cities, so I avoided route and city one-hot encoding in the final feature set.

---

**[1:10–1:35] Feature engineering and model choice**

My final feature set is called Q: distance, log distance, distance bins, cleaned weight with a missing flag, quote signal, and equipment.

I tested richer feature sets with route, geography, and market index, but the simpler Q set generalized better.

The final model is HistGradientBoosting with a log-one-plus target transform — train on log of rate, predict with expm1 back to dollars.

---

**[1:35–2:00] Validation strategy and results**

I used chronological validation, not random shuffling, to simulate predicting future time periods.

Primary split: train January through August, validate September and October.

Sensitivity split: train through September, validate October only.

On development validation, my final model achieved MAE of one hundred six point eight three on the primary split and one hundred thirteen point two nine on sensitivity.

Important: these are development validation results on labeled data — not the final November–December holdout score, because those labels aren't available locally.

---

**[2:00–2:25] Final model and predictions**

For final training, I fit on all forty-eight thousand January–October labeled rows.

Then I generated twelve thousand predictions for the holdout set in the required format: load ID and predicted rate.

Prediction mean is about two thousand three hundred forty-six dollars, median about two thousand twenty-six — broadly aligned with training.

All predictions are positive and finite.

---

**[2:25–2:45] score.py and December chart**

I ran the official `score.py` script, which validates submission format and generates a December chart.

It does not compute MAE or any accuracy metric — Spotter evaluates that externally after submission.

score.py exited successfully. It validated twelve thousand predictions and thirty-one December rows, and created the candidate December chart.

The chart is flat at eight hundred forty-one dollars because my locked feature set doesn't include calendar features — only the date changes in that fixed Lexington to Fort Wayne scenario.

---

**[2:45–3:00] Limitations and conclusion**

Remaining limitations: long-haul and extreme high-rate loads are still underpredicted, CatBoost wasn't available to test, and I can't score Nov–Dec accuracy locally.

In summary: HistGradientBoosting with feature set Q and log target, strong chronological validation, twelve thousand holdout predictions generated, and score.py validation passed.

Thanks for watching.
