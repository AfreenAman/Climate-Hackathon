# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:43:44 2022

@author: dreji18
"""

from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import components
from GoogleNews import GoogleNews
from gnews import GNews
from arcgis.gis import *
import datetime
from datetime import date
from dateutil import parser

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pydeck as pdk
from geopy.geocoders import Nominatim


@st.cache()
def load_world_port_index_data():
    wpid_df = pd.read_excel("World Port Index.xlsx")
    return wpid_df

@st.cache(allow_output_mutation=True)
def news_extractor(search_text):
    google_news = GNews()
    json_resp = google_news.get_news(search_text)
    if len(json_resp) < 10:
        return pd.DataFrame()
    else:
        article_data = []
        for link_id in range(0, 10):
            try:
                url = json_resp[link_id]['url']
                article_obj = google_news.get_full_article(url)
                if article_obj == None:
                    article = None
                    article_data.append(article)
                else:
                    article = article_obj.text
                    article_data.append(article)
            except:
                article = None
                article_data.append(article)
                continue
        
        news_df = pd.DataFrame(json_resp)[0:10]
        news_df['article'] = article_data
        return news_df

@st.experimental_singleton
def semantic_model():
    # initialize the sentence transformers model
    #return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device='cuda')
    return SentenceTransformer('multi-qa-MiniLM-L6-cos-v1',device='cpu')

@st.cache()
def relevant_embedding(content, embed_model, filtered_words_list):
    """embedding representation of text"""
    embeddings_sentences = embed_model.encode(content) 
    embeddings_dictionary = embed_model.encode(['\n'.join(filtered_words_list)])
    similarity_matrix = cosine_similarity(embeddings_dictionary, embeddings_sentences) 
    content_relevance = pd.DataFrame(content, columns = ["Content"])
    content_relevance['relevance'] = similarity_matrix.T
    content_relevance = content_relevance.sort_values(by=['relevance'], ascending=False).reset_index(drop=True) 

    return content_relevance

@st.cache()
def date_fetching():
 
    # news_df['published date'] = news_df['published date'].astype(str)
    # news_df['published date'] = news_df['published date'].apply(lambda x: parser.parse(x, fuzzy=True).date())
    
    # start_date = min(news_df['published date'])
    # end_date = max(news_df['published date'])      

    start_date = datetime.date(2021, 1, 1)
    end_date = date.today()
    
    return start_date, end_date

def content_fetching(news_df, start_date, end_date): 
    try:
        df = news_df[news_df['published date'].between(start_date, end_date)]    
    except:
        df = news_df
    df = df[df['article'].notna()] 
    
    all_para = []
    for para in df['article']:
        int_list = para.split("\n\n")
        for i in int_list:
            if len(i.split()) > 5 and len(i.split()) < 500:
                if '?' not in i[-3:]:
                    all_para.append(i)
    
    return all_para

@st.cache()
def news(region):
    try:
        loc = Nominatim(user_agent="GetLoc")
        getLoc = loc.geocode(region)
        Lat = getLoc.latitude
        Lon = getLoc.longitude
        
        geolocator = Nominatim(user_agent="geoapiExercises")
        location = geolocator.reverse(str(Lat)+","+str(Lon))
        address = location.raw['address']
        country = address.get('country', '')
            
        googlenews = GoogleNews(period='100d')
        googlenews.get_news(country + ' Marine Shipping news')
        result=googlenews.result()
    except:
        result = []
        Lat = None
        Lon = None
    return result, Lat, Lon

def main():
    # page details
    st.set_page_config(page_title='Web_Results',layout= 'wide', page_icon='🖖')
    wallpaper1 = Image.open('wallpaper cropped.png')
    wallpaper1 = wallpaper1.resize((1200,300)) 
    st.sidebar.image(wallpaper1)
    st.sidebar.info("This app has been developed as part of Climate Hackathon 2022 organised and hosted by Macsimum in partnership with Microsoft")
    
    wallpaper2 = Image.open('Image20220608115804.png')
    wallpaper2 = wallpaper2.resize((1400,500))      
    st.image(wallpaper2)
    
    st.write(" ")
    st.markdown("_Based on the location selected by the user, this app provides information related to shipping activities and their effect on environment. It also raise awareness among residents and tourists about the condition of the oceans. This app uses NLP language models to extract the information_")
    
    # loading the main data
    wpid_df = load_world_port_index_data()
    
    # options sidebar
    options = ("Overview", "Ocean App")
    value = st.sidebar.radio("Select the Option", options,0)
    
    if value == "Ocean App":
        st.sidebar.subheader("🎲 Web Search")
        
        st.subheader("🎲 Shipping Port Pollution Impact Assessment")
        
        #st.dataframe(wpid_df.head())
        text_query = st.sidebar.selectbox("Enter the location you prefer", 
                                  wpid_df['Country Code'].unique())
        
        embed_model = semantic_model()
        news_df = news_extractor(str(text_query + " planned ports"))
        start_date, end_date = date_fetching()
        
        st.sidebar.subheader("Select the time range")
        start_date = st.sidebar.date_input('Start date', start_date)
        end_date = st.sidebar.date_input('End date', end_date)
        if start_date > end_date:
            st.error('Error: End date must fall after start date.')
        
        if len(news_df) == 0:
            st.warning("Sorry no data available or failed to extract!!!")
        else:
            #start_date, end_date = date_fetching(news_df)
            
            content_data = content_fetching(news_df, start_date, end_date)            
            question_list = ["What are the planned shipping ports in the region?"]
            
            with st.expander(question_list[0], expanded=True):
                filtered_words_list = ["shipping ports"]
                try:
                    planned_ports = relevant_embedding(content_data, embed_model, filtered_words_list)
                    st.write('\n\n'.join(planned_ports['Content'][0: 3].to_list()))
                except:
                    st.write("No data found")
                
                st.write('\n')
                st.markdown("**Current number of Ports in this location : " +str(len(wpid_df[wpid_df['Country Code'] == text_query]['Main Port Name'].unique())) + "**")
            
        news_df1 = news_extractor(str(text_query + " ocean biodiversity"))  
        if len(news_df1) == 0:
            st.warning("Sorry no data available or failed to extract!!!")
        else:
            content_data1 = content_fetching(news_df1, start_date, end_date)
            question_list1 = ["What are the effects of Shipping port activities on Ocean Biodiversity?"]
            
            with st.expander(question_list1[0]):
                filtered_words_list = ["ocean biodiversity", "marine life", "aquatic ecosystem"]
                try:
                    ocean_biodiversity = relevant_embedding(content_data1, embed_model, filtered_words_list)
                    st.write('\n\n'.join(ocean_biodiversity['Content'][0: 3].to_list()))
                except:
                    st.write("No data found")
            
        news_df2 = news_extractor(str(text_query + " ocean pollution")) 
        if len(news_df2) == 0:
            st.warning("Sorry no data available or failed to extract!!!")
        else:
            content_data2 = content_fetching(news_df2, start_date, end_date)
            question_list2 = ["What is the impact of the ports on marine pollution?"]
            
            with st.expander(question_list2[0], expanded=True, ):
                filtered_words_list = ["ocean pollution", "marine pollution"]
                try:
                    pollution = relevant_embedding(content_data2, embed_model, filtered_words_list)
                    st.write('\n\n'.join(pollution['Content'][0: 3].to_list()))
                except:
                    st.write("No data found")
                
                garbage_df = wpid_df[wpid_df['Country Code'] == text_query]
                
                st.bar_chart(garbage_df['Garbage Disposal'])
                
                st.bar_chart(garbage_df['Chemical Holding Tank Disposal'])
                
                st.bar_chart(garbage_df['Dirty Ballast Disposal'])
                
        news_df3 = news_extractor(str(text_query + " ship accidents"))
        if len(news_df3) == 0:
            st.warning("Sorry no data available or failed to extract!!!")
        else:
            content_data3 = content_fetching(news_df3, start_date, end_date)
            question_list3 = ["what is the Risk of accidents due to shipping ports?"]
            
            with st.expander(question_list3[0], expanded=True):
                filtered_words_list = ["shipping accidents", "oil spillage", "ship collision", "maritime accidents"]
                try:
                    accidents = relevant_embedding(content_data3, embed_model, filtered_words_list)
                    st.write('\n\n'.join(accidents['Content'][0: 3].to_list()))
                except:
                    st.write("No data found")
        
        news_df4 = news_extractor(str(text_query + " ship fuel"))
        if len(news_df4) == 0:
            st.warning("Sorry no data available or failed to extract!!!")
        else:
            content_data4 = content_fetching(news_df4, start_date, end_date)
            question_list4 = ["what is the fuel used at the port?"]
            
            with st.expander(question_list4[0]):
                filtered_words_list = ["diesel", "pertrol", "fuels"]
                try:
                    fuel = relevant_embedding(content_data4, embed_model, filtered_words_list)
                    st.write('\n\n'.join(fuel['Content'][0: 3].to_list()))
                except:
                    st.write("No data found")
                
                fuel_df = wpid_df[wpid_df['Country Code'] == text_query]
                st.bar_chart(fuel_df['Services - Electricity'])
                
                st.bar_chart(fuel_df['Supplies - Fuel Oil'])
                
                st.bar_chart(fuel_df['Supplies - Diesel Oil'])
                
                st.bar_chart(fuel_df['Supplies - Aviation Fuel'])
        
        news_df5 = news_extractor(str(text_query + " ship green fuel"))
        if len(news_df5) == 0:
            st.warning("Sorry no data available or failed to extract!!!")
        else:
            content_data5 = content_fetching(news_df5, start_date, end_date)
            question_list5 = ["Are the ports switching to greener fuels?"]
            
            with st.expander(question_list5[0]):
                filtered_words_list = ["hydrogen", "ammonia", "green fuel", "electric"]
                try:
                    alternate_fuel = relevant_embedding(content_data5, embed_model, filtered_words_list)
                    st.write('\n\n'.join(alternate_fuel['Content'][0: 3].to_list()))
                except:
                    st.write("No data found")
              
        # geolocation map
        try:
            loc_df = wpid_df[wpid_df['Country Code'] == text_query]
            df = loc_df[['Latitude', 'Longitude']]
            latest_news, text_latitude, text_longitude = news(text_query) #returns latest news as well as locations latitude and logitude
            
            # if (text_latitude != None) and (text_longitude != None):
            #     df.loc[len(df)] = [text_latitude, text_longitude, text_query]
            
            layer = pdk.Layer(
                "ScatterplotLayer",
                df,
                pickable=True,
                opacity=0.8,
                filled=True,
                radius_scale=2,
                radius_min_pixels=10,
                radius_max_pixels=500,
                line_width_min_pixels=0.01,
                get_position='[Longitude, Latitude]',
                get_fill_color=[255, 0, 0],
                get_line_color=[0, 0, 0],
            )
            
            # Set the viewport location
            view_state = pdk.ViewState(latitude=df['Latitude'].iloc[-1], longitude=df['Longitude'].iloc[-1], zoom=5, min_zoom= 0, max_zoom=50)
            
            # Render, mapbox street view
            r = pdk.Deck(layers=[layer], map_style='mapbox://styles/mapbox/streets-v11',
                         initial_view_state=view_state, tooltip={"html": "<b>Point ID: </b> {PointID} <br /> "
                                                                         "<b>Longitude: </b> {Longitude} <br /> "
                                                                         "<b>Latitude: </b>{Latitude} <br /> "
                                                                         "<b> Value: </b>{Value}"})
            r 
        except:
            st.warning("Sorry, location plot cannot be displayed!")

        # showing news headlines about the shipping events in the place recently
        st.subheader("🎲 Latest Regional Shipping News")
        cols = st.columns(2)
        #latest_news = news(text_query)
        if latest_news == []:
            st.write("No Recent news captured!")
        else:
            counter = 0
            for i in latest_news[0:6]:
                if counter in [0, 2, 4]:
                    j = 0
                else:
                    j = 1
                
                cols[j].write(i['title'])
                cols[j].image(
                    i['img'],
                    width=100, # Manually Adjust the width of the image as per requirement
                )
                link_ = i['link']
                link_ = link_.replace("news.google.com/.", "https://news.google.com") 
                link_path = "read this [article]" + "(" + link_ + ")"
                cols[j].write(link_path)
                cols[j].write(" ")
                counter+=1

    if value == "Overview":
        st.subheader("Introduction/Background")
        st.write("The environmental effects of marine transport include air pollution, water pollution, acoustic, and oil pollution. The demand for seaborne trade is projected to grow by 39% until 2050. Concepts like “blue corridors” (critical ocean habitats for migratory marine species), “green transportation” are surfacing as potential solutions to enable early adoption of alternative fuels and conservation of vulnerable ocean areas. We have developed a solution that makes it easier to gain insight into marine pollution and how particularly vulnerable areas and marine life are affected.")
        st.write("Our app gives insights and raises citizen awareness on the impact of shipping ports and how vulnerable areas and marine life are affected. It gives current news on accidents (oil spill etc.) and helps tourists avoid their travel, it helps residents avoid activities like fishing in the surrounding region.")
        st.write("Our long-term vision is to raise citizen awareness and collective consciousness on the pollution impact of ports, make all users of our app contributors to creating a database and help governments and regulatory bodies device measures to curtail port based negative environmental impact with information collection and assessment.")
        
        
        
        with st.expander("About the Authors !!!", expanded=True):
            col1, col2 = st.columns([4,8])
            with col1:  
                st.image("IMG_20190413_092554.jpg")
                st.write(" ")
                st.write(" ")
            with col2:   
                st.markdown("**Afreen Aman**")
                st.write("An Environmental data scientist who works towards integrating data science and data analytics solutions in environment, climate change, sustainability, and sustainable finance. She is currently working on developing and managing sustainable digital solutions using NLP for ESG and GHG data analysis. She has developed AI solutions/ prototypes for sustainable finance. She has participated in various national and international Conferences for poster and paper presentations and has published papers in international journals. She has Co-Trained BERT on environmental data and hosted a python package: “EnvBert” on PyPI")
                st.write(" ")
                #st.write(" ")
    
            with col1:  
                st.image("Deepak Jr_main photo1.jpg")
            with col2:       
                st.markdown("**Deepak John Reji**")
                st.write("An NLP practitioner with experience in developing and designing solutions for data science products. He loves working with Environmental & Natural Assets data, and creates videos on prototypes that he experiments with, NLP tutorials, and Podcasts with industry experts in the Sustainability and AI Industry. He is an open-source contributor. Recently he has been researching on the topics “Bias & Fairness in AI Models” and “AI in Environment Due Diligence” where he has developed a package called “Dbias” and co -trained a model named “EnvBert” on environmental data, respectively. ")
                    
                
# calling the main function
if __name__ == "__main__":
    main()