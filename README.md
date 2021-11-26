# Final Project - KICKSTARTER

![picture](kickstarter.png)

Kickstarter is a funding platform for creative projects. Everything from film, games, and music to art, design, and technology. Kickstarter is full of ambitious, innovative, and imaginative projects that are brought to life through the direct support of others.

Every project creator sets their project's funding goal and deadline. If people like the project, they can pledge money to make it happen. If the project succeeds in reaching its funding goal, all backers' credit cards are charged when time expires. Funding on Kickstarter is all-or-nothing. If the project falls short of its funding goal, no one is charged.

## Objective

The objective is to predict if a project will receive it's goal funds.

## Data

### Data Source

The data has been downloaded from https://webrobots.io/kickstarter-datasets/

### Shape of the Data

Before cleaning the dataframe consisted of 543.589 rows and 39 columns. 
After cleaning the dataframe used for EDA and model consisted of 186.303 rows and 1.012 columns.

Here are the columns that were used for the final model:

| Column name | Description |
| ----------- | ----------- |
| goal | amount to be fully funded |
| state | successful (1) or failed (0) project |
| category_slug | name of category and subcategory concatenated |
| category_parent_name | name of category |
| launched_at_weekday | weekday the project was launched |
| deadline_weekday | weekday the project was finished |
| project_duration | duration of project |
| blurb_language_new | description language grouped |
| country_new | countries grouped |
| description_words | count of words |
| description_sent | count of sentences |
| descr_stopw | count of stopwords |
| description_filtered_words | count of filtered words |
| desc_filt_ratio | ratio of filtered words per description words |
| desc_sw_ratio | ratio of stopwords per description words |
| desc_wps_ratio | ratio or words per sentence |
| description_neg | sentiment analysis from nltk library - negative words |
| description_neu | sentiment analysis from nltk library - neutreal words |
| description_pos | sentiment analysis from nltk library - positive words |
| description_compound | sentiment analysis from nltk library - compound of words |
| columns xxx to xxx | encoding of 1.000 most common words in successful projects |

## Python Libraries
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scipy](https://www.scipy.org/)
- [sklearn](https://scikit-learn.org/stable/)
- [imblearn](https://imbalanced-learn.org/stable/)
- [textblob](https://textblob.readthedocs.io/en/dev/)
- [langid](https://pypi.org/project/langid/1.1.2dev/)
- [nltk](https://www.nltk.org/)