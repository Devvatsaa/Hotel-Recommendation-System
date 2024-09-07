import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



df= pd.read_csv("HotelRecommend_final.csv")

tfidf=TfidfVectorizer(max_features=1000)


def recommend(type, country, city , property ,starrating,  user_id= None):
    # Check if user_id matches from historical data
    if user_id:
        user_history = get_user_history(user_id, df1)
        if not user_history.empty:
            user_tags = ' '.join(user_history['tags'].tolist())
        else:
            user_tags = f"{type} {country} {city} {property} {starrating}"
    else:
        user_tags = f"{type} {country} {city} {property} {starrating}"

    # Filter the dataset based on the country and city
    temp = df[(df['country'] == country) & (df['city'] == city) & (df['starrating'] >= starrating) &
            (df['roomtype'] == type) & (df['propertytype'] == property) ]
     # Append user preferences to the filtered DataFrame

    if temp.empty: #an empty list is returned if no hotels matches the criteria
        return pd.DataFrame()

    # Create a DataFrame from user tags
    user_tags_df = pd.DataFrame({'tags': [user_tags]})
    
    temp = pd.concat([temp, user_tags_df],ignore_index=True)
    
    # Fit and transform the TF-IDF vectorizer
    vector = tfidf.fit_transform(temp['tags']).toarray()
    user_index = len(temp)-1
    # Calculate cosine similarity matrix
    similarity = cosine_similarity(vector)
    
    # Get indices of the filtered hotels
    filtered_indices = temp[temp['tags'] == user_tags].index.tolist()
    
    # Recommend top 5 similar hotels for each filtered hotel

    similar_hotels = sorted(list(enumerate(similarity[user_index])), key=lambda x: x[1],reverse= True)[1:6]
    # Skip the first match (itself)
    recommended_hotels=[]
    for hotel in similar_hotels:
            #print(tuple(temp.loc[hotel[0]][['hotelname', 'roomtype','starrating']]))
        hotel_details=temp.loc[hotel[0]][['hotelname', 'roomtype','starrating','url']]
        recommended_hotels.append(hotel_details)
    #recommended_hotels = [temp.iloc[i[0]]['hotelname'] for i in similar_hotels]
        
    
    return pd.DataFrame(recommended_hotels)

st.title(":blue[Hotel Recommender System]")

st.header(":rainbow[Enter your preference as per your choice]")

room_type = st.selectbox(
'Roomtype(The kind of room you want)',
df['roomtype'].unique()
)

country = st.selectbox(" Select Country", df['country'].unique())

city = st.selectbox("Select City", df['city'].unique())

property_type = st.selectbox(
'Select Property Type',df['propertytype'].unique()
)

starrating = st.slider('Select Minimum Star Rating', 1, 5, 3)

if st.button('Get Recommendations'):
    recommendations = recommend(room_type, country, city, property_type, starrating)
    
    #if recommendations:
       # st.write('Recommended Hotels:')
        #for hotel in recommendations:
         #   st.write(f"*Hotel Name:* {hotel['hotelname']}")
          #  st.write(f"*Room Type:* {hotel['roomtype']}")
           # st.write(f"*Star Rating:* {hotel['starrating']}")
            #st.write(f"[*Book Now*]({hotel['url']})")
            #t.write("---")
    if not recommendations.empty:
        st.write('Recommended Hotels:')
        
        # Adding a clickable URL in the DataFrame
        recommendations['url'] = recommendations['url'].apply(lambda x: f'<a href="{x}" target="_blank">Book Now</a>')
        
        # Set up the Streamlit table with clickable URLs
        st.write(recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write('No hotels found matching your preferences.')