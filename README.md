# ML Final Project

## Group Members

Ned Cutler and Maddy Smith

## Abstract (TODO)

In 3-4 sentences, concisely describe:

- What problem your project addresses.
- The overall approach you will use to solve the problem.
- How you propose to evaluate your success against your stated goals.

## Motivation and Question (TODO)

Describe your motivation for your project idea. Some (shortened) examples of good types of motivations:

- We have a scientific data set for which predictive or exploratory models would help us generate hypotheses.
- We have user information for which predictive models would help us give users better experiences.
- We have performance data (e.g. from sports teams) for which predictive models could help us make better decisions.
- Algorithmic bias is an increasingly urgent challenge as machine learning products proliferate, and we want to explore it more deeply.
- You should be more specific than these: describe your specific data set (if applicable); your scientific questions; the type of decisions your model could inform; etc.

## Planned Deliverables

**Full success:**
- A python package containing a complete pipeline: data collection, feature engineering, and model training/evaluation.
- Custom torch implementations of logistic regression and a feedforward neural network, with no use of scikit-learn or other pre-built model libraries. Possibly other models as well, like a CNN.
- A jupyter notebook demonstrating the full pipeline, including exploratory data analysis, model comparisons, and visualizations.

**Partial success:**
- Even if the more powerful models don't outperform logistic regression, which is plausible with our slightly smaller dataset, our deliverable should still include the full data pipeline, the logistic regression baseline with experiments, and an analysis of why the simpler model wins.

## Resources Required

**Data sources (all freely accessible):**

1. Open-Meteo Historical Weather API (open-meteo.com/en/docs/historical-weather-api): Free access to ERA5 reanalysis weather data at any lat/lon coordinate going back to 1940. Returns daily temperature, precipitation, snowfall, snow depth, wind speed, humidity in JSON format.

2. bestsnow.net Vermont Snow Conditions Chart (bestsnow.net/vrmthist.htm): A table of weekly condition grades (A/B/C/D/R) for Vermont ski conditions, covering 26 seasons (1999-2000 through 2024-25), yielding ~650 labeled data points. Grades are based on terrain openness and snow surface quality (powder/packed powder vs. hardpack/ice vs. rain). 

**Compute:** Laptops should be fine, colab possibly. I (Ned) also have a PC with an Nvidia GPU that I can use if needed.

## What You Will Learn (TODO)

In this section, please state what each group member intends to learn through working on the project. You might be thinking of certain algorithms, software packages, version control, project management, effective teamwork, etc.

### Ned

I want to deepen my understanding of implementing neural networks from scratch in torch, particularly around the practical challenges of training on small datasets (regularization, class imbalance, overfitting). I also want to learn how to build a clean data pipeline that scrapes, cleans, and aligns data from multiple sources and formats. On the ML theory side, I'm interested in exploring when neural networks actually help over logistic regression and whether ordinal structure in the target variable can be exploited.

### Maddy

## Risk Statement

1. Small dataset size (~650 samples) may not be enough for neural network approaches. With only 26 seasons of weekly labels, overfitting is a serious concern, and the feedforward neural network may not outperform logistic regression.

2. The bestsnow.net condition grades are curated by a single analyst and may contain inconsistencies over a 26-year span. The grading methodology may have shifted subtly over time, and the grades reflect expert-skier preferences (valuing off-piste powder) that don't generalize to all skier types. If the labels are too noisy, no model will achieve strong performance.

## Ethics Statement

1. **Overall impact:** If successful and deployed, this project would give skiers a weather-based tool for independently assessing likely ski conditions without relying on resort-reported snow reports, which have an inherent marketing bias. This promotes transparency and more informed consumer decision-making.

2. **Who benefits:** Skiers and snowboarders in Vermont would benefit from more unbiased conditions forecasts. 

3. **Who is is excluded/harmed:** Resorts could potentially be harmed if poor-snow predictions keep skiers away on marginal days. Our data also comes from expert skier preference, so it could be less applicable to beginner/intermediate skiers. Additionally, the dataset covers mainly Vermont's largest, highest snowfall resorts, so predictions might not generalize to smaller areas like the Snow Bowl.

4. **Will the world become an overall better place:** Yes, if:
   - Skiers make better decisions (safer driving choices, more satisfying trips, etc.) when they have accurate, unbiased condition information.
   - The net effect of better informed skiers does not cause a net reduction in overall ski visits, but a redistribution towards days with better conditions.

## Tentative Timeline

**Weeks 9-10 (by ~April 22):** 
- Data acquisition and exploratory analysis complete. 
- Scrape bestsnow.net labels, pull Open-Meteo weather data for all six resort coordinates across all 26 seasons, build the feature engineering pipeline (weekly aggregation), and align labels with features. 
- Produce EDA notebook with class distribution plots, feature correlation analysis, and a visual exploration of the 26-season condition trend. 
- Deliverable for checkpoint: a working dataset and initial logistic regression baseline with preliminary accuracy numbers.

**Weeks 10-11 (by ~April 29):** 
- Implement feedforward neural network in PyTorch. 
- Run full experiment suite: feature ablation, ordinal vs. categorical framing, lagged condition features, class imbalance handling. 
- Possibly implement other models like CNN. 
- Begin writing final blog post and preparing presentation.

**Week 12 (by ~May 3):** 
- Complete analysis, finalize blog post, clean up GitHub repository, deliver presentation.