"""
Module for topic modelling, so we can plug in 
either FRUS or Sasha's text predictions

Password file contains credentials to log in to db. Not committed.
"""
import passwd
import re

def get_texts(start_date='1958-01-01', end_date='1970-01-01', limit=None):
    """Establish database connection and 
    perform query to fetch rows
    
    Parameters
    ----------
    start_date : limit query by this start date;
                format YYYY-MM-DD
    end_date : limit query by this end date;
                format YYYY-MM-DD
    limit : limit the number of documents to fetch
    
    Returns
    -------
    all_texts : list of strings, body of each doc
    """    
    db = passwd.get_frus_db()
    cursor = passwd.get_cursor(db)
    # Total number of rows in docs is 26933
    
    q = "select body \
            from docs \
            where date >= \
            date('" + start_date + "') \
            and date <= \
            date('" + end_date + "')"

    if limit:
        q += " limit " + str(limit)	

    cursor.execute(q)
    all_texts = [row['body'] for row in cursor]

    return all_texts

def get_redactions(all_texts):
    """Format: list of lists, list per document of redactions"""
    redact = [re.findall("\[.*?declassified.*?\]", text) \
            for text in all_texts]
    return redact

def get_indices(all_texts):
    """Indices of redactions"""
    indices = [re.finditer("\[.*?declassified.*?\]", text) \
            for text in all_texts]
    return indices

def get_para(all_texts):
    """Get the paragraph surrounding the redaction 
        including the redaction marking.
        returns lists of list per document"""
    para = [re.findall("\n(.*?\[.*?declassified.*?\].*?)\n", text, re.MULTILINE) \
            for text in all_texts]
    return para

def get_surround(all_para):
    """TODO: doesn't successfully remove partial markups
        Note that this also flattens the list, 
        Now it's a list of strings not of lists."""
    surr = [re.sub("\[.*?declassified.*?\]", "", p, re.MULTILINE) \
            for doc in all_para \
            for p in doc]
    return surr
 
def flatten(l):
    """Flatten a list of lists"""
    return [item for sublist in l for item in sublist]
    
########### Old declassification.frus table ##################
def get_texts_old(limit=None):
    """Establish database connection and 
    perform query to fetch rows
    
    Parameters
    ----------
    limit : limit the number of documents to fetch
    
    Returns
    -------
    cursor : pointer to the DB
    """    
    db = passwd.get_db()
    cursor = passwd.get_cursor(db)
    if limit:
        q = "select fullbody from frus limit " + str(limit) 
    else:
        q = "select fullbody from frus"
    cursor.execute(q)
    all_texts = [row['fullbody'] for row in cursor]

    return all_texts


