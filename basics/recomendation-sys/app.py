import pickle
import streamlit as st
import requests



headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0OGQ0MWNhNjQxN2MxZjFhYTFjN2U4ZGMxOTA1YmM3OSIsIm5iZiI6MTc2Njg3NzYyMy4zOTM5OTk4LCJzdWIiOiI2OTUwNjliNzAyNmM0Y2JlNDM4OGQ4MjkiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.H1dgIlCfx13lDdWMLb3gViFVj0GsrETWZDO3s9oiNn8",
    "accept": "application/json"
}

params = {
    "include_adult": "false",
    "language": "en-US",
    "page": 1
}

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    response = requests.get(url, headers=headers)
    data = response.json()

    poster_path = data.get('poster_path')
    if poster_path:
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    return None



def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    
    distance = sorted(
        list(enumerate(similarity[index])),
        key = lambda x: x[1],
        reverse=True, 
    )

    recommended_movies_name =[]
    recommended_movies_poster =[]

    for i in distance[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies_poster.append(fetch_poster(movie_id))
        recommended_movies_name.append(movies.iloc[i[0]].title)

    return recommended_movies_name, recommended_movies_poster



st.header("Movie Recommendation system using Machine Learning")
movies = pickle.load(open('artifacts/movie_list.pkl', 'rb'))
similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))



print(similarity)


movies_list = movies['title'].values
selected_movie = st.selectbox("Type or select a movie to get a recommendation",
             movies_list
             )

if st.button("Show recommendation"):
    recommended_movies_name , recommended_movies_poster = recommend(selected_movie)
    col1, col2, col3,col4,  col5 = st.columns(5)
    with col1:
        st.text(recommended_movies_name[0])
        st.image(recommended_movies_poster[0])

    with col2:
        st.text(recommended_movies_name[1])
        st.image(recommended_movies_poster[1])

    with col3:
        st.text(recommended_movies_name[2])
        st.image(recommended_movies_poster[2])

    with col4:
        st.text(recommended_movies_name[3])
        st.image(recommended_movies_poster[3])
    
    with col5:
        st.text(recommended_movies_name[4])
        st.image(recommended_movies_poster[4])