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
- Custom torch implementations of logistic regression and a feedforward neural network, with no use of scikit-learn or other pre-built model libraries. Possibly other models, like a CNN
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

## Ethics Statement (TODO)

All projects we undertake involve decisions about whose interests matter; which problems are important; and which tradeoffs are considered acceptable. Take some time to reflect on the potential impacts of your project on its prospective users and the broader world. Address the following questions:

1. What is the overall impact that we would hope to have on the world if our project was successful and deployed?

2. What groups of people have the potential to benefit from our project?

3. What groups of people have the potential to be excluded from benefit or even harmed from our project?

4. Will the world become an overall better place because we made our project? Describe at least 2 assumptions behind your answer. For example, if your project aims to make it easier to predict crime, your assumptions might include:

   - Criminal activity is predictable based on other features of a person or location.
   - The world is a better place when police are able to perform their roles more efficiently.

If your project involves making decisions or recommendations, then you will also need to consider possible forms of algorithmic bias in your work. Here are some relevant examples:

- A recipe recommendation app can privilege the cuisines of some locales over others. Will your user search recipes by ingredients? Peanut butter and tomato might seem an odd combination in the context of European cuisine, but is common in many traditional dishes of the African diaspora. A similar set of questions applies to recommendation systems related to style or beauty.

- A sentiment analyzer must be trained on specific languages. What languages will be included? Will diverse dialects be included, or only the “standard” version of the target language? Who would be excluded by such a choice, and how will you communicate about your limitations?

## Tentative Timeline (TODO)

We will have a checkpoint for the project in Week 9 or 10, and then final presentations in Week 12. With this in mind, please describe what you expect to achieve after two and four weeks. Your goal by the two-week check-in should be to have “something that works.” For example, maybe in two weeks you’ll ready to demonstrate the data acquisition pipeline and show some exploratory analysis, and in the last couple weeks you’ll actually implement your machine learning models.