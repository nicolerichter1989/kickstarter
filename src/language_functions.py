## add libraries

from textblob import TextBlob
import langid
import ast
import yake
from deep_translator import GoogleTranslator
from collections import OrderedDict, defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer
from string import punctuation


## functions

#
#
#
def add_language_column(column, df):

    '''this function returns a column identifying the language of the input column''' 

    language = []   

    for i in df[f'{column}']:
        if isinstance(i, float):
            language.append(0)
        else:
            a = langid.classify(i)
            language.append(a[0])

    df[f'{column}' + '_language'] = language

    return df
#
#
#
def language_translation_english(column, df):

    '''this function returns the translation to english of text for the input column'''
    
    trans = []

    for i in df[f'{column}']:
        if isinstance(i, float):
            trans.append(0)
        else:
            a = GoogleTranslator(source='auto', target='en').translate(i)
            trans.append(a)
    
    df[f'{column}' + '_trans'] = trans

    return df
#
#
#
def language_keywords(column, df):

    '''this function returns keywords for the input column'''

    r = Rake()

    kw = []

    for i in df[f'{column}']:
        if isinstance(i, float):
            kw.append(0)
        else:
            r.extract_keywords_from_text(i)
            kw.append(r.get_ranked_phrases_with_scores())
    
    df[f'{column}' + '_kw'] = kw

    return df
#
#
#
def translation(column, df):

    '''this function returns the translation to english of text for the input column'''

    trans = []

    for i in df[f'{column}']:
        if isinstance(i, float):
            trans.append('')
        else:
            if len(i) >= 5000:
                trans.append('')
            else:
                a = GoogleTranslator(source='auto', target='en').translate(i)
                trans.append(a)
    
    df[f'{column}' + '_trans'] = trans

    return df
#
#
#
def nltk_sentiment(column, df):

    '''this function returns different nltk sentiments for the input column'''
    
    sia = SentimentIntensityAnalyzer()

    neg = []
    neu = []
    pos = []
    compound = []

    for i in df[f'{column}']:

        if isinstance(i, float):
            neg.append('')
            neu.append('')
            pos.append('')
            compound.append('')  
       
        else:
            dictionary = sia.polarity_scores(i)
    
            neg.append(dictionary.get('neg'))
            neu.append(dictionary.get('neu'))
            pos.append(dictionary.get('pos'))
            compound.append(dictionary.get('compound'))


    df[f'{column}' + '_neg'] = neg
    df[f'{column}' + '_neu'] = neu
    df[f'{column}' + '_pos'] = pos
    df[f'{column}' + '_compound'] = compound

    return df
#
#
#
def group_columns(column, df, threshold):

    '''this function groups the input column based on the threshold input'''
    
    a = df[f'{column}'].value_counts()
    dd = defaultdict(list)
    languages = a.to_dict(dd)

    new = []

    for i in df[f'{column}']:

        if languages.get(i) <= threshold:
            new.append('other')
        else:
            new.append(i)

    df[f'{column}' + '_new'] = new

    return df
#
#
#
def more_nltk(column, df):

    '''this function returns different nltk extracts from the input column'''

    stop_words = set(stopwords.words('english'))

    words = []
    filtered_words = []
    sent = []
    stopw = []

    for i in df[f'{column}']:

        if isinstance(i, float):
            words.append('')
            filtered_words.append('')
            sent.append('')
            stopw.append('')
        
        else:
            words.append(len(word_tokenize(str(i))))
            filtered_words.append(len([w for w in i if not w.lower() in stop_words]))
            sent.append(len(sent_tokenize(str(i))))
            stopw.append(len([w for w in i if w in stop_words]))

    df[f'{column}' + '_words'] = words
    df[f'{column}' + '_filtered_words'] = filtered_words
    df[f'{column}' + '_sent'] = sent
    df[f'{column}' + '_stopw'] = stopw

    return df
#
#
#
def get_keyword_base(column, df):

    '''this function returns the kw base'''
    from string import punctuation
    df[f'{column}'] = df[f'{column}'].apply(str)
    stop_words = set(stopwords.words('english'))
    punctuation = list(punctuation)

    kw_base  = []

    for i in df[f'{column}']:
        lower_tokens = [x.lower() for x in word_tokenize(i)]
        final_tokens = [t for t in lower_tokens if t not in stop_words and t not in punctuation]
        kw_base.append(final_tokens)    

    df[f'{column}' + '_kw_base'] = kw_base

    return df
#
#
#
def get_final_keywords(column, df):

    '''this function returns the keywords'''
    from string import punctuation
    df[f'{column}'] = df[f'{column}'].apply(str)
    stop_words = set(stopwords.words('english'))
    punctuation = list(punctuation)

    r = Rake()

    keywords  = []

    for i in df[f'{column}']:
        lower_tokens = [x.lower() for x in word_tokenize(i)]
        final_tokens = [t for t in lower_tokens if t not in stop_words and t not in punctuation]
        r.extract_keywords_from_text(final_tokens)
        scores = r.get_word_degrees()
        keywords.append(scores.keys())

    df[f'{column}' + '_kw_final'] = keywords

    return df
#
#
#
