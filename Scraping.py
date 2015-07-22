
# coding: utf-8


# ## Name: Akilesh R. Student ID.1001091662
# ## Name: Ashwin R. Student ID.1001098716 
# ## Name: Anirudh R. Student ID.1001051262



# importing re for Walmart Sccrapping
import re


# special IPython command to prepare the notebook for matplotlib
get_ipython().magic(u'matplotlib inline')

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#A sane way to get web data
import requests

#Packages for web scraping. No need to use both. Feel free to use one of them.
from pattern import web
from bs4 import BeautifulSoup

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns

#For some of the date operations
import datetime


# ### Website 1: Wikipedia
# -----------------------
# 
# 
# 
# 

# In[4]:

#Input:
#    url: URL of a wikipedia page
#    table_name: Name of the table to scrape
#Output:
#    df is a Pandas data frame that contains a tabular representation of the table
#    The columns are named the same as the table columns
#    Each row of df corresponds to a row in the table
def get_headers(rows):
    """ Get header data from rows """
    
    results = []
    table_headers = rows.findAll("th")
    if table_headers:
        results.append([headers.get_text() for headers in table_headers])
        
    table_data = rows.findAll("td")
    if table_data:
        results.append([data.get_text() for data in table_data])
    
    return results

def get_data(rows):
    """ Get data from rows """
    
    results = []
    for row in rows:
        table_headers = row.findAll("th") +  row.findAll("td")
        results.append([headers.get_text() for headers in table_headers])
    return results

def scraping_wikipedia_table(url, table_num):
    """ 
    Input:
    url: URL of a wikipedia page
    table_num: Number of the table
    Output:
    Pandas dataframe containing the data found on the wiki page
    """
    try:
        r = requests.get(url)
        
    except URLError as e:
        print 'An error occured fetching %s \n %s' % (url, e.reason)   
        return 1
    
    bs = BeautifulSoup(r.content)
    #print bs
    tableNum = int(table_num) - 1
    try:
        tables = bs.findAll('table', 'wikitable')
        tblGiven = tables[tableNum]
        
    except IndexError as e:
        print "No valid Table found at number",table_num
        return 1
    
    rows = tblGiven.findAll("tr")
    
    table_header = get_headers(rows[0])
    rows.pop(0)
    table_data = get_data(rows)
    df = pd.DataFrame(table_data, columns=table_header[0])
    table_len = len(df) - 1
    wiki_dataframe = df.drop(table_len)
    return wiki_dataframe
print scraping_wikipedia_table("http://en.wikipedia.org/wiki/List_of_Test_cricket_records", 17)


# 
# ### Website 2: Walmart
# 

# In[ ]:

#Input:
#    url: URL of a Walmart results page for a search query in Movies department
#Output:
#    df is a Pandas data frame that contains a tabular representation of the results
#    The df must have 9 columns that must have the same name and data type as described above
#    Each row of df corresponds to a movie in the results table
def scraping_walmart_movies(url):
    r = requests.get(url)
    bs = BeautifulSoup(r.text)
    timecheck = 0
    notime = 0
    tableMain = []
    pricelist = []
    runningtime = []
    ratings = []
    formatlist = []
    castlist = []
    titlelist = []
    ratinglist = []
    shiplist = []
    pickuplist = []
    for movies in bs.findAll('div','tile-content'):
        #movie title
        title = movies.findAll('a','js-product-title')
        titleval = title[0].get_text()
        titlelist.append(str(titleval))
    
        #movie price
        price = movies.find('span','price','price-display')
        if price is None:
            noprice = 0
            pricelist.append(float(noprice))
        else:
            priceval = re.sub('[^0-9.]+','',price.text)
            pricelist.append(float(priceval))
        
        cast = movies.find('dt','media-details-starring')
        if cast is None:
            castval = "No actors found"
            castlist.append(str(castval))
        else:
            index3 = 0
            for castvalcheck in movies.findAll('dt'):
                index3 = index3 + 1
                if castvalcheck.text == 'Starring:':
                    castdata = movies.findAll('dd')
                    finalactors = castdata[index3 - 1].text
                    castlist.append(str(finalactors))
            
            
        #rating of the movie
        rating = movies.find('span','js-reviews')
        if rating is None:
            norating = 0
            ratings.append(int(norating))   
        else:
            editedrate = re.sub('[^0-9.]+','',rating.text)
            ratings.append(int(editedrate[0]))
        
        #duration of the movie
        timing = movies.find('dt','media-details-running-time')
        if timing is None:
            ftime = 0
            runningtime.append(int(ftime))       
        else:
            index = 0
            for titlecheck in movies.findAll('dt'):
                index = index + 1
                if titlecheck.text == 'Running:':
                    ftime = movies.findAll('dd')
                    movietime = re.sub('[^0-9]+','',ftime[index - 1].text)
                    runningtime.append(int(movietime))
            
            
        #format of the screen
        format = movies.find('dt','media-details-format')
        if format is None:
            finalformatvalue = "No format found"
            formatlist.append(str(finalformatvalue))
        else:
            index2 = 0
            for formatcheck in movies.findAll('dt'):
                index2 = index2 + 1
                if formatcheck.text == 'Format:':
                    formatvalue = movies.findAll('dd')
                    finalformatvalue = formatvalue[index2 - 1].text
                    formatlist.append(str(finalformatvalue))    
            
        #rating of the movie
        rrating = movies.find('dt','media-details-rating')
        if rrating is None:
            finalrratingvalue = "Not Rated"
            ratinglist.append(str(finalrratingvalue))
        else:
            index4 = 0
            for rratingcheck in movies.findAll('dt'):
                index4 = index4 + 1
                if rratingcheck.text == 'Rating:':
                    rratingvalue = movies.findAll('dd')
                    finalrratingvalue = rratingvalue[index4 - 1].text
                    ratinglist.append(str(finalrratingvalue))    
            
        #shipping and Pickup
        ship = True
        pickup = True
        shipping = movies.find('ul','block-list','fullfillment-container')
        if shipping is None:
            ship = False
            pickup = False
            shiplist.append(ship)
            pickuplist.append(pickup)
        else:
            formatship = re.sub('[^0-9A-Za-z]+','',shipping.text)
            if (formatship == 'Freeshippingonordersover50Freestorepickuptoday') or (formatship == 'Freeshippingonordersover50Freestorepickup') or (formatship == 'FreeshippingFreestorepickup'):
                ship = True
                pickup = True
            elif (formatship == 'Freestorepickuptoday') or (formatship == 'Freestorepickup'):
                ship = False
            elif (formatship == 'Freeshippingonordersover50'):
                pickup = False
            else:
                ship = False
                pickup = False
            shiplist.append(ship)
            pickuplist.append(pickup)    
    tableMain = zip(titlelist, pricelist, ratings, shiplist, pickuplist, castlist, runningtime, formatlist, ratinglist)

    df = pd.DataFrame(data=tableMain,columns = ['Product Title','Sale Price','Number of Ratings','Free Shipping','Free Store Pickup','Starring','Running','Format', 'Rating'])
    return df
            


# ###  Website 3: Facebook
# 
# 

# In[ ]:

def scraping_facebook_books(dom):
    """
        Input: dom - DOM of the books page
        Output: An array (Python list) of books listed in the profile page.
    """
    
    book_name = []
    for booksGroup in dom.findAll("ul", "uiList _620 _14b9 _5pst _5psx _509- _4ki"):
        books = booksGroup.findAll("li","_5rz")
    for book in books:
        book_name.append(book.find("div","_gx6 _agv").find('a').get_text())
    #print book_name
    return book_name

def scraping_facebook_groups(dom):
    """
        Input: dom - DOM of the groups page 
        Output: A Pandas data frame with one row per group.
    """
    group_name = []
    group_members = []
    group_description = []
    table_header = ['Group Name', 'Number of Members', 'Group Description']
    for Groups in dom.findAll("ul", "uiList _4-sn _509- _4ki"):
        group = Groups.findAll("li","_1v6c")
        
        for group_item in group:
            group_name.append(group_item.find("div","mbs fwb").find('a').get_text())
        
            group_members.append(group_item.find("div","mbs fcg").get_text())
        
            g_desc = group_item.find("span","_538r").get_text()
            if not g_desc:
                group_description.append("No Description available")
            else:
                group_description.append(g_desc)
        
    
    group_info = zip(group_name,group_members,group_description)
    df = pd.DataFrame(group_info, columns=table_header)
    #print df 
    return df

def scraping_facebook_music(dom):
    """
        Input: dom - DOM of the music page
        Output: A Pandas data frame with one row per group. 
    """
    
    music_name = []
    music_type = []
    is_verified = []
    profile_url = []
    music_header = ['Name', 'Type', 'Verified', 'Profile Url']
    for music_pages in dom.findAll("ul", "uiList _620 _14b9 _5psw _509- _4ki"):
        music_page = music_pages.findAll("li","_5rz")
    for music in music_page:
        music_name.append(music.find("div","_gx6 _agv").find('a').get_text())
        
        music_type.append(music.find("div","_1fs8 fsm fwn fcg").get_text())
        
        verify_music = music.find("span","_56_f _5dzy _5dzz")
        if verify_music is None:
            is_verified.append("False")
        else:
            is_verified.append("True")
    
        profile_url.append(music.find("div","_gx6 _agv").find('a').get('href'))
    
    music_table = zip(music_name,music_type,is_verified,profile_url)
    df = pd.DataFrame(music_table, columns=music_header)
    #print df
    return df

def scraping_facebook_movies(dom):
    """
        Input: dom - DOM of the movies page
        Output: A Pandas data frame with one row per movie. 
    """
    movie_name = []
    movie_type = []
    is_verified = []
    profile_url = []
   
    movie_header = ['Name', 'Type', 'Verified', 'Profile Url']
    #movie_header = ['Name', 'Type', 'Verified', 'Profile Url', 'Likes', 'Starring', 'Genre', 'Director', 'Movie URL']
    
    for movie_pages in dom.findAll("ul", "uiList _620 _14b9 _5pst _5psw _509- _4ki"):
        movie_page = movie_pages.findAll("li","_5rz")
    for movie in movie_page:
        movie_name.append(movie.find("div","_gx6 _agv").find('a').get_text())
        
        movie_type.append(movie.find("div","_1fs8 fsm fwn fcg").get_text())
        
        verify_movie = movie.find("span","_56_f _5dzy _5dzz")
        if verify_movie is None:
            is_verified.append("False")
        else:
            is_verified.append("True")
    
        profile_url.append(movie.find("div","_gx6 _agv").find('a').get('href'))
    
    movie_table = zip(movie_name,movie_type,is_verified,profile_url)
    #print movie_table
    df = pd.DataFrame(movie_table, columns=movie_header)
    return df 

