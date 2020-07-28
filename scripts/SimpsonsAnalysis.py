#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.font_manager as fm
from scipy.signal import argrelextrema
from string import capwords
import os

episodesdf = pd.read_csv('../data/simpsons_episodes.csv')
linesdf = pd.read_csv('../data/simpsons_script_lines.csv')

#Create output directory
try:
    if not os.path.isdir('../output'):
        os.mkdir('../output')
except OSError:
    print('Creation of output folder failed')
else:
    print('Successfully created output directory')

#Remove stage directions
linesdf = linesdf[linesdf['speaking_line']==True]
linesdf.drop(columns=['id', 'episode_id', 'number', 'timestamp_in_ms', 'speaking_line', 'character_id', 'location_id', 'raw_text', 'spoken_words', 'normalized_text'], inplace=True)
linesdf.head()


#Format season codes to SXXEXX
episodesdf['season'] = episodesdf['season'].apply(lambda x: 'S'+str(x).zfill(2))
episodesdf['number_in_season'] = episodesdf['number_in_season'].apply(lambda x: 'E'+str(x).zfill(2))
episodesdf['code'] = episodesdf[['season', 'number_in_season']].agg(''.join, axis=1)
episodesdf.drop(columns=['id', 'number_in_season', 'number_in_series', 'original_air_year', 'production_code', 'season', 'video_url', 'views', 'imdb_votes'], inplace=True)
episodesdf.head()

print('Number of unique speakers is ', len(np.unique(linesdf['raw_character_text'].tolist())))


#General formatting
linesdf['raw_character_text'] = linesdf['raw_character_text'].astype(str)
linesdf['raw_location_text'] = linesdf['raw_location_text'].astype(str)
linesdf = linesdf[linesdf['word_count'].astype(str).str.isdigit()]
linesdf['word_count'] = linesdf['word_count'].astype(int)
linesdf['raw_location_text'] = linesdf['raw_location_text'].apply(lambda x: capwords(x))
linesdf['raw_character_text'] = linesdf['raw_character_text'].apply(lambda x: capwords(x))
linesdf.head()


#Group by lines spoken by character and their location
linesdf = linesdf.groupby(['raw_character_text', 'raw_location_text'], as_index=False).sum()
linesdf.head()


#Group by character speech
speechdf = linesdf.drop(columns=['raw_location_text'])
speechdf = speechdf.groupby(['raw_character_text'], as_index=False).sum()


speechdf.sort_values(by='word_count', inplace=True, ignore_index=True, ascending=False)
speechdf.head()



speechdf = speechdf.head(20)



#Create pie chart
pie = px.pie(speechdf, values='word_count', names='raw_character_text', title='Spoken Lines of Simpsons Characters')
pie.write_html('../output/simpsonslinesspoken.html')



episodesdf.sort_values(by='original_air_date', inplace=True, ignore_index=True)


#Use scipy to get local max/min
ratings = np.array(episodesdf['imdb_rating'])
max_indices = np.array(argrelextrema(ratings, np.greater, order=15))[0]
min_indices = np.array(argrelextrema(ratings, np.less, order=15))[0]




fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111)

ax.plot(np.array(episodesdf.index)+1, episodesdf['imdb_rating'], 'r-')

ax1 = ax.twinx()

ax1.plot(np.array(episodesdf.index)+1, episodesdf['us_viewers_in_millions'], 'b-')
ax.set_xlabel('Episode Number', fontsize=20)
ax.set_ylabel('IMDb Rating', fontsize=20, color='r')
ax1.set_ylabel('Viewers (in millions)', fontsize=20, color='b', rotation=-90, labelpad=20)

ax.tick_params(labelsize=20)
ax.xaxis.set_ticks(np.arange(0, episodesdf.shape[0]+1, 25))
ax1.tick_params(labelsize=20)

#Label mins
for i in min_indices:
    ax.annotate(episodesdf.iat[i,episodesdf.shape[1]-1], (i, episodesdf.iat[i, 1]), xytext= (i+1, episodesdf.iat[i, 1]-0.15), arrowprops=dict(arrowstyle='->'), fontsize=10)

#Label maxes
for i in max_indices:
    ax.annotate(episodesdf.iat[i,episodesdf.shape[1]-1], (i, episodesdf.iat[i, 1]), xytext = (i+1, episodesdf.iat[i, 1]+0.15), arrowprops=dict(arrowstyle='->'), fontsize=10)


simp_logo = mpimg.imread('../media/SimpsonsLogo.png')
simp_family = mpimg.imread('../media/Simpsons_FamilyPicture.png')

#[x position, y position, x size, y size]
imax = fig.add_axes([0.8, 0.8, 0.2, 0.2], anchor = 'NE')
imax1 = fig.add_axes([0.015, 0.875, 0.125, 0.125], anchor = 'NW')
imax.imshow(simp_logo)
imax1.imshow(simp_family)
imax.axis('off')
imax1.axis('off')

#Replace font
font = fm.FontProperties(fname='../media/simpsonfont/Simpsonfont-p07r.ttf', size = 40)

ax1.set_title('Simpsons Ratings and Viewers over Time', fontproperties=font)

plt.savefig('../output/SimpsonsRatings.png', dpi=221)



#Pivot data
linespivotdf = linesdf.pivot(index='raw_character_text', columns='raw_location_text', values='word_count')
linespivotdf = linespivotdf.fillna(0)
linespivotdf.head()



linespivotdf['sum'] = linespivotdf.sum(axis=1)
linespivotdf = linespivotdf.sort_values(by='sum', ascending=False).drop(columns=['sum'])
linespivotdf.head()



#Sort columns by sum and take largest 10
linespivotdf = linespivotdf.reindex(linespivotdf.sum().sort_values().index, axis=1)
linespivotdf = linespivotdf.iloc[:, -10:]
linespivotdf.head()



linespivotdf = linespivotdf.head(10)
linespivotdf.head(10)



#Get proportion of lines spoken
linespivotdf = linespivotdf.apply(lambda x: x/x.sum()*100)
linespivotdf = linespivotdf.T
linespivotdf.head(10)



fig1 = plt.figure(figsize=(20, 20))
ax1 = fig1.add_subplot(111)

font1 = fm.FontProperties(fname='../media/simpsonfont/Simpsonfont-p07r.ttf', size = 40)


ax1.set_ylabel('Percent of Lines Spoken', fontsize=20)
ax1.set_title("Simpsons' Characters Lines Spoken by Location", fontproperties=font1)
linespivotdf.plot(ax=ax1, kind='bar', stacked=True)

box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax1.legend(loc='center right', bbox_to_anchor = (1.25,0.5), frameon=False, title='Characters', title_fontsize=15, fontsize=15, labelspacing=2.5)
ax1.set_xlabel('Simpsons Locations', fontsize=20)
ax1.set_xticklabels(labels=linespivotdf.index.values.tolist(), rotation=30, fontsize=12.5)
ax1.set_yticklabels(labels=np.arange(0,101,20), fontsize=12.5)

imax2 = fig1.add_axes([0.775, 0.79, 0.2, 0.2], anchor = 'NE')
imax2.imshow(simp_logo)
imax2.axis('off')

#Add images sequentially
simpsons_chars = ['hs_head.png', 'ls_head.png', 'ms_head.png', 'bs_head.png', 'mb_head.png', 'mo_head.png', 'ss_head.png', 'nf_head.png', 'kk_head.png', 'cw_head.png']
directory = '../media/'
placement = 0.63
for head in simpsons_chars:
    char_head = mpimg.imread(directory+head, 0)
    imax3 = fig1.add_axes([0.9, placement, 0.025, 0.025])
    imax3.imshow(char_head)
    imax3.axis('off')
    placement-=0.035

ax1.grid(True, linestyle='dashed', linewidth=0.5)
fig1.savefig('../output/SimpsonsSpeech.png', dpi=221)

