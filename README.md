# Final Project - KICKSTARTER

![picture](kickstarter.png)

Kickstarter is a funding platform for creative projects. Everything from film, games, and music to art, design, and technology. Kickstarter is full of ambitious, innovative, and imaginative projects that are brought to life through the direct support of others.

Every project creator sets their project's funding goal and deadline. If people like the project, they can pledge money to make it happen. If the project succeeds in reaching its funding goal, all backers' credit cards are charged when time expires. Funding on Kickstarter is all-or-nothing. If the project falls short of its funding goal, no one is charged.

## Objective

The goal  is to be able to predict whether or not a project will receive it's goal funds.

## Data

describe data etc.

### Data Source

The data has been extracted via apify downloads from kickstarter.

### Shape of the Data

Before cleaning the dataframe consisted of 543.589 rows and 39 columns. 
After cleaning the dataframe used for EDA and model consisted of 176.721 rows and 22 columns.

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
| blurb_language_new | language of description grouped |
| xxx | xxx |
| columns xxx to xxx | encoding of 1.000 most common words in successful projects |


### Notes from Tableau EDA

- project duration ranges from x to y
- project duration is shorter for successful projects (among all categories, countries)
- the average goal is lower for more successful projects
- no country except japan has a successful project over 5M
- generally most successful project reached a goal up to 1M
- almost all project over 1M fail
- the failed% increases as the goal increases

### What makes a project successful?

- goal under 5k 67% success - between 5k and 100k still 51%
- successful project averagely last 32 days
- categories technology, photography, food, crafts and journalism succeed less often (-50%)
- a project described in english is more succeeds more often then e.g. es,fe,de,it,nl
- the most successful projects are launched and finished on a tuesday (73,48%)
- generally most successful projects are launched on tuesdays
- 




## Python Libraries
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scipy](https://www.scipy.org/)
- [sklearn](https://scikit-learn.org/stable/)

### Libraries use for NLP
- [textblob](https://textblob.readthedocs.io/en/dev/)
- [yake](https://pypi.org/project/yake/)
- [langid](https://pypi.org/project/langid/1.1.2dev/)
- [rake-nltk](https://pypi.org/project/rake-nltk/)