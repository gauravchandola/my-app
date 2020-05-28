#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:57:26 2020

This program creates a dash web application for the DeepFly project
 
@author: gauravchandola
"""
#from dash.react import Dash
#
#my_app = Dash('my app')
 
# -*- coding: utf-8 -*-
#import RecSys_Cont as rc
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output, State, ALL
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


colors = {
    'background': '#EEE',
    'text': '#7FDBFF'
}

with open('Final_DB.json','r') as f:
    DF_db = json.load(f)
    
    
    
My_db = pd.DataFrame(list(DF_db.items()),columns=['Name','Description'])
My_db['Description'] = My_db['Description'].apply(' '.join)

#np.save('cosine_similarities.npy', cosine_similarities )
cosine_similarities=np.load('cosine_similarities.npy')

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(My_db['Description'])



app.layout = html.Div(children=[


    html.Div(style={'backgroundColor': colors['background']}, children=
    html.H1(children='DeepFly', 
            style={
            'textAlign': 'center',
            'color': '#35BF19'
        }              
     )),
    
    
    html.Div(children=html.H5('''
         A travel recommendor system - Taking care of your travel needs
    ''', style={
            'backgroundColor': colors['background'],
            'textAlign': 'center',
            'color': colors['text']
        }
    )),
    

            
     html.Label('How many places have you been to:'),
        dcc.Input(id='m_input', value='0', type='text'),
        html.Button("Submit", id="submit", n_clicks=0),
        
     html.Div(id='my-div', children=[]),
     
     html.Button("Submit", id="submit2", n_clicks=0),
     
     html.Div(id='output2', children=[]),
        
     html.Button("Ready", id="submit3", n_clicks=0),
     
     html.Div(id='output3', children=[])
        
    
 ])   
        
 
@app.callback(
        Output(component_id='my-div', component_property='children'),
        [Input(component_id='submit', component_property='n_clicks')],
        [State('m_input', 'value')]
)

def display_input(n_clicks, value):
    children=[]
    for i in range(1, int(value)+1):
        new_label= html.Label('Enter place no {}:'.format(i))
        new_input = dcc.Input(
                id={ 'type': 'text-input',
                     'index': i},
                value='0',
                type='text'
                )
        children.append(new_label)
        children.append(new_input)
    
    return children


@app.callback(
        Output(component_id='output2', component_property='children' ),
        [Input(component_id='submit2', component_property='n_clicks')],
        [State(component_id={'type': 'text-input', 'index': ALL}, component_property='value')]
        
    
        
        )

#def display_output2(POI):
#     POI = POI.lower()
#     if POI in DF_db.keys():
#         return 'Success! Enter next place'          
#     else:       
#         return  'Fail! please enter another place'
    
def display_output(n_clicks,values):
    n=len(values)
    children=[]
    for (i, value) in enumerate(values):
        POI=value.lower()
        if POI in DF_db.keys():
            if i==0:
                children.append(html.Div('Success! {} is in the database. '.format(value)))
            else:
                children.append(html.Div('\n Success! {} is in the database. \n'.format(value)))
        else:       
            children.append(html.Div('\n Fail! {} is not in the database. Enter another place'.format(value)))
            break
        
        if i==n-1:
            children.append(html.Br())
            children.append(html.Div('Ready for your recommendations??!!  '))
#            children.append(html.Button("Read", id="submit3", n_clicks=0))
            
    return children



@app.callback(
        Output(component_id='output3', component_property='children'),
        [Input(component_id='submit3', component_property='n_clicks')],
        [State(component_id={'type': 'text-input', 'index': ALL}, component_property='value')]
    
        )

def recommendations(n_click, values):
    
    children=[]
    
    def find_index(poi,db):
        for idx, row in db.iterrows():
            if(poi == row[0]):
                return idx
                break
        return -1
    
    poi_index = []
    
    for POI in values:
        POI=POI.lower()
        poi_index.append(find_index(POI,My_db))
    
    
    for i, idx in enumerate(poi_index):
        if i == 0:
            usr_prof = tfidf_matrix[idx,:]
        else:
            usr_prof = usr_prof + tfidf_matrix[idx,:]
        usr_prof = usr_prof/5.0
    
    # Compute similarity scores for the user profile
    SIM =np.squeeze(linear_kernel(usr_prof,tfidf_matrix)) 
    ind_sim = SIM.argsort()# used for sorting the output of the sim.argsort function
    ind_sim = ind_sim[::-1]
    # Recommend
    N = 10 # top N places
    count = 0
    i = 0
#    print('Top Recommendations: \n')
    
    while (count <N):
        idx = ind_sim[i]
        if idx in poi_index: # already given by user
            i = i+1
        else:
            poi_name=My_db.loc[idx]['Name'].upper()
            poi_sim=round(SIM[idx],2)
            children.append(html.Div(' {} with score of = {}'.format(poi_name, poi_sim)))
            count = count + 1
            i = i + 1
    
    return children
        

    
if __name__ == '__main__':
    app.run_server(debug=True)