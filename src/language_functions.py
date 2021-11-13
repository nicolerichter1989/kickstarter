## add libraries

from textblob import TextBlob
import langid
import yake
from rake_nltk import Rake
import nltk
from deep_translator import GoogleTranslator

## functions

def all_text_and_language_functions(column, df):

    '''this function returns all transformations (language, polarity, subjecticity, text length, keywords) to the input column''' 

    language_features(column, df)

    language_length(column, df)

    language_keywords(column, df)

    return df
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
def language_features(column, df):

    '''this function returns language features polarity and subjectiviy for the input column''' 

    polarity = []
    subjectivity = []

    for i in df[f'{column}']:
        if isinstance(i, float):
            polarity.append(0)
            subjectivity.append(0)
        else:
            a = TextBlob(i)
            polarity.append(round(a.polarity,2))
            subjectivity.append(round(a.subjectivity,2))

    df[f'{column}' + '_pol'] = polarity
    df[f'{column}' + '_sub']  = subjectivity

    return df
#
#
#
def language_length(column, df):

    '''this function returns the length of text for the input column'''

    length = []

    for i in df[f'{column}']:
        if isinstance(i, float):
            length.append(0)
        else:    
            length.append(len(i))
    
    df[f'{column}' + '_len'] = length

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

    '''this function returns the length of text for the input column'''

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

    '''this function returns the length of text for the input column'''

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

