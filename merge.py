# Import the libraries
#coding=utf-8
import os
import pandas as pd
import numpy as np
import json
import seaborn as sns
import re
#%config InlineBackend.figure_format ='retina'
import spotipy
import matplotlib.pyplot as plt
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import random
from pandas.io.json import json_normalize

# Declare the credentials


cid ='65c9880bd14140b38615f33d3afe884a' 
secret ='4e5c388390884e81b0ac756ef17fadc4'
redirect_uri='http://localhost:8887/callback'
username = 'Dddd'

# Authorization flow

scope = 'user-top-read'
token = util.prompt_for_user_token(username, scope, client_id=cid, client_secret=secret, redirect_uri=redirect_uri)

if token:
    sp = spotipy.Spotify(auth=token)
else:
    print("Can't get token for", username)


# Fetch the top-50 songs of the user (short term)
if token:
    sp = spotipy.Spotify(auth=token)
    resultss = sp.current_user_top_tracks(limit=50,offset=0,time_range='short_term')
    for song in range(50):
        list1 = []
        list1.append(resultss)
        with open('short_term.json', 'w', encoding='utf-8') as f:
            json.dump(list1, f, ensure_ascii=False, indent=4)
else:
    print("Can't get token for")
# Open the JSON file to Python objects
with open('short_term.json',encoding='utf-8') as f:
  data1 = json.load(f)

# Fetch the top-50 songs of the user (medium term)
if token:
    sp = spotipy.Spotify(auth=token)
    resultsm = sp.current_user_top_tracks(limit=50,offset=0,time_range='medium_term')
    for song in range(50):
        list1 = []
        list1.append(resultsm)
        with open('medium_term.json', 'w', encoding='utf-8') as f:
            json.dump(list1, f, ensure_ascii=False, indent=4)
else:
    print("Can't get token for")
    # Open the JSON file to Python objects
with open('medium_term.json',encoding='utf-8') as f:
  data2 = json.load(f)

#合併兩個json檔案
resultstotal = data1[0]['items'] + data2[0]['items']

#取得合併後的內容
list_of_resultstotal= resultstotal
list_of_artist_names = []
list_of_artist_uri = []
list_of_song_names = []
list_of_song_uri = []
list_of_durations_ms = []
list_of_explicit = []
list_of_albums = []
list_of_popularity = []

for result in list_of_resultstotal:
    result["album"]
    this_artists_name = result["name"]
    list_of_artist_names.append(this_artists_name)
    
    this_artists_uri = result["uri"]
    list_of_artist_uri.append(this_artists_uri)
    
    list_of_songs = result["name"]
    list_of_song_names.append(list_of_songs)
    
    song_uri = result["uri"]
    list_of_song_uri.append(song_uri)
    
    list_of_duration = result["duration_ms"]
    list_of_durations_ms.append(list_of_duration)
    
    song_explicit = result["explicit"]
    list_of_explicit.append(song_explicit)
    
    this_album = result["album"]["name"]
    list_of_albums.append(this_album)
    
    song_popularity = result["popularity"]
    list_of_popularity.append(song_popularity)

# Convert the pulled content to a pandas df

all_songs = pd.DataFrame(
    {'artist': list_of_artist_names,
     'artist_uri': list_of_artist_uri,
     'song': list_of_song_names,
     'song_uri': list_of_song_uri,
     'duration_ms': list_of_durations_ms,
     'explicit': list_of_explicit,
     'album': list_of_albums,
     'popularity': list_of_popularity
     
    })
#Making into A CSV

all_songs.to_csv('song_merge_term.csv') 

usersongsm = pd.read_csv('song_merge_term.csv')
usersongss = pd.read_csv('song_merge_term.csv')
temp = [usersongsm, usersongss]
temp = pd.concat(temp)
temp.reset_index(drop=True,inplace=True)

userm_list = []
for song in usersongsm['song_uri']:
    row = pd.DataFrame(sp.audio_features(tracks=[song]))
    userm_list.append(row)
userm_df = pd.concat(userm_list)

users_list = []
for song in usersongss['song_uri']:
    row = pd.DataFrame(sp.audio_features(tracks=[song]))
    users_list.append(row)
users_df = pd.concat(users_list)

# Combine both users' top 50 songs into one dataframe of 100 songs

dfs = [userm_df, users_df]
dfs = pd.concat(dfs)

dfs.drop(['type','track_href','analysis_url','time_signature','duration_ms','uri','instrumentalness','liveness','loudness','key','mode'],1,inplace=True)
dfs.set_index('id',inplace=True)

# Normalize tempo feature

columns = ['danceability','energy','speechiness','acousticness','valence','tempo']
scaler = MinMaxScaler()
scaler.fit(dfs[columns])
dfs[columns] = scaler.transform(dfs[columns])

# Get 20 clusters from 100 songs

clusters = 20
kmeans = KMeans(n_clusters=clusters)

from sklearn.decomposition import PCA  

pca = PCA(3)  

from matplotlib import colors as mcolors 
import math 
   
''' Generating different colors in ascending order  
                                of their hsv values '''
colors = list(zip(*sorted(( 
                    tuple(mcolors.rgb_to_hsv( 
                          mcolors.to_rgba(color)[:3])), name) 
                     for name, color in dict( 
                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS 
                                                      ).items())))[1] 
   
   
# number of steps to taken generate n(clusters) colors  
skips = math.floor(len(colors[5 : -5])/clusters) 
cluster_colors = colors[5 : -5 : skips] 

scaler = MinMaxScaler()
scaled = scaler.fit_transform(dfs)
y_kmeans = kmeans.fit_predict(scaled)

# Updating dataframe with assigned clusters 

dfs['cluster'] = y_kmeans
dfs['artist'] = temp.artist.tolist()
dfs['title'] = temp.song.tolist()

# Removing clusters that only have one song in them

deleteclusters = []
cluster = 0
while cluster < (len(dfs.cluster.unique())-1):
    if dfs.groupby('cluster').count().loc[cluster].danceability == 1:
        deleteclusters.append(cluster)
    cluster+=1

dfs.reset_index(inplace=True)
dfs.set_index('id',inplace=True)

# 把要放入推薦歌曲function的歌曲id列表放入一個新列表中
i=0
list_of_recs = [0]*len(dfs.groupby('cluster').count())
while i<len(dfs.groupby('cluster').count()):
    list_of_recs[i] = dfs.loc[dfs['cluster'] == i].index.to_list()
    i+=1
    
list_of_recs = [ele for ele in list_of_recs if ele != []] 

# Adjust list for clusters so that each cluster has a maximum of 5 seed songs

j = 0
adj_list_of_recs = [0]*len(list_of_recs)
while j<len(list_of_recs):
    if 0 < len(list_of_recs[j]) < 6:
        adj_list_of_recs[j] = list_of_recs[j]
    elif len(list_of_recs[j]) > 5:
        adj_list_of_recs[j] = random.sample(list_of_recs[j], 5)
    j += 1

#Getting 1 recommended song from each cluster with less than 4 songs, 2 recommended songs from each cluster with 4-5 songs

k = 0
list_of_recommendations = [0]*len(list_of_recs)
while k < len(list_of_recs):
    if len(adj_list_of_recs[k]) < 4:
        list_of_recommendations[k] = sp.recommendations(seed_tracks=adj_list_of_recs[k],limit=1)
    else:
        list_of_recommendations[k] = sp.recommendations(seed_tracks=adj_list_of_recs[k],limit=2)
    k += 1
list_of_recommendations_converted = [0]*len(list_of_recs)

l = 0
while l < len(list_of_recs):
    list_of_recommendations_converted.append(pd.json_normalize(list_of_recommendations[l], record_path='tracks').id.tolist())
    l += 1

no_integers = [x for x in list_of_recommendations_converted if not isinstance(x, int)]
list_of_recommendations_converted = [item for elem in no_integers for item in elem]

print(list_of_recommendations_converted)