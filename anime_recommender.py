import requests
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

class AnimeRecommender:
    def __init__(self):
        self.base_url = "https://api.jikan.moe/v4"
        self.anime_data = None
        self.genre_matrix = None
        self.vectorizer = CountVectorizer()
        
    def search_anime(self, query, limit=5):
        """
        Busca animes basados en un término de búsqueda
        """
        endpoint = f"{self.base_url}/anime"
        params = {
            "q": query,
            "limit": limit
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for anime in data.get('data', []):
                anime_info = {
                    'id': anime.get('mal_id'),
                    'title': anime.get('title'),
                    'image_url': anime.get('images', {}).get('jpg', {}).get('image_url'),
                    'synopsis': anime.get('synopsis'),
                    'score': anime.get('score'),
                    'genres': [genre.get('name') for genre in anime.get('genres', [])]
                }
                results.append(anime_info)
                
            return results
        except requests.exceptions.RequestException as e:
            print(f"Error al buscar anime: {e}")
            return []
    
    def get_anime_details(self, anime_id):
        """
        Obtiene detalles detallados de un anime específico
        """
        endpoint = f"{self.base_url}/anime/{anime_id}"
        
        try:
            response = requests.get(endpoint)
            response.raise_for_status()
            anime = response.json().get('data', {})
            
            anime_details = {
                'id': anime.get('mal_id'),
                'title': anime.get('title'),
                'image_url': anime.get('images', {}).get('jpg', {}).get('image_url'),
                'synopsis': anime.get('synopsis'),
                'score': anime.get('score'),
                'genres': [genre.get('name') for genre in anime.get('genres', [])],
                'studios': [studio.get('name') for studio in anime.get('studios', [])],
                'year': anime.get('year'),
                'episodes': anime.get('episodes'),
                'rating': anime.get('rating'),
                'themes': [theme.get('name') for theme in anime.get('themes', [])]
            }
            
            return anime_details
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener detalles del anime: {e}")
            return None
    
    def fetch_top_anime(self, limit=25):
        """
        Obtiene los animes mejor calificados para construir nuestro dataset inicial
        Nota: La API Jikan puede tener limitaciones en el parámetro limit
        """
        endpoint = f"{self.base_url}/top/anime"
        params = {
            "limit": min(limit, 25)  # La API puede tener un límite máximo, usamos 25 para evitar errores
        }
        
        try:
            anime_list = []
            # Hacemos múltiples solicitudes si limit > 25
            for page in range(1, (limit // 25) + 2):
                if len(anime_list) >= limit:
                    break
                    
                params["page"] = page
                
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                
                for anime in data.get('data', []):
                    if len(anime_list) >= limit:
                        break
                        
                    anime_info = {
                        'id': anime.get('mal_id'),
                        'title': anime.get('title'),
                        'image_url': anime.get('images', {}).get('jpg', {}).get('image_url'),
                        'synopsis': anime.get('synopsis'),
                        'score': anime.get('score'),
                        'genres': [genre.get('name') for genre in anime.get('genres', [])],
                        'year': anime.get('year')
                    }
                    anime_list.append(anime_info)
                
                # Respetar el límite de tasa de la API
                time.sleep(1)
                
            self.anime_data = pd.DataFrame(anime_list)
            return self.anime_data
        except requests.exceptions.RequestException as e:
            print(f"Error al obtener los mejores animes: {e}")
            return pd.DataFrame()
    
    def prepare_genre_matrix(self):
        """
        Prepara una matriz para comparar similitudes basadas en géneros
        """
        if self.anime_data is None or self.anime_data.empty:
            print("No hay datos de anime disponibles. Ejecute fetch_top_anime primero.")
            return
        
        # Crear una columna con géneros concatenados para el vectorizador
        self.anime_data['genres_string'] = self.anime_data['genres'].apply(lambda x: ' '.join(x) if x else '')
        
        # Crear una matriz de características de género
        genre_matrix = self.vectorizer.fit_transform(self.anime_data['genres_string'])
        self.genre_matrix = genre_matrix
        
        return genre_matrix
    
    def get_recommendations(self, anime_id=None, anime_title=None, top_n=5):
        """
        Obtiene recomendaciones basadas en un anime específico por ID o título
        """
        if self.anime_data is None or self.genre_matrix is None:
            print("Datos no preparados. Ejecute fetch_top_anime y prepare_genre_matrix primero.")
            return []
        
        # Encontrar el índice del anime en nuestro dataset
        if anime_id:
            anime_idx = self.anime_data[self.anime_data['id'] == anime_id].index
        elif anime_title:
            anime_idx = self.anime_data[self.anime_data['title'].str.contains(anime_title, case=False)].index
        else:
            return []
        
        if len(anime_idx) == 0:
            return []
        
        anime_idx = anime_idx[0]
        
        # Calcular similitud de coseno
        cosine_sim = cosine_similarity(self.genre_matrix)
        
        # Obtener puntuaciones de similitud con otros animes
        sim_scores = list(enumerate(cosine_sim[anime_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Obtener los índices de los animes más similares (excluyendo el propio anime)
        sim_scores = sim_scores[1:top_n+1]
        anime_indices = [i[0] for i in sim_scores]
        
        # Devolver los animes recomendados
        recommendations = self.anime_data.iloc[anime_indices].to_dict('records')
        return recommendations
    
    def recommend_by_genres(self, genres, top_n=5):
        """
        Recomienda animes basados en una lista de géneros
        """
        if self.anime_data is None:
            print("No hay datos de anime disponibles. Ejecute fetch_top_anime primero.")
            return []
            
        # Filtrar animes que contienen al menos uno de los géneros especificados
        if not genres:
            return []
            
        filtered_animes = self.anime_data[
            self.anime_data['genres'].apply(lambda x: any(genre in x for genre in genres))
        ]
        
        # Ordenar por puntuación y devolver los N primeros
        top_animes = filtered_animes.sort_values('score', ascending=False).head(top_n)
        return top_animes.to_dict('records')
    
    def get_anime_by_year(self, year, top_n=5):
        """
        Obtiene los mejores animes de un año específico
        """
        if self.anime_data is None:
            print("No hay datos de anime disponibles. Ejecute fetch_top_anime primero.")
            return []
            
        filtered_animes = self.anime_data[self.anime_data['year'] == year]
        top_animes = filtered_animes.sort_values('score', ascending=False).head(top_n)
        return top_animes.to_dict('records')
    
    def get_popular_genres(self):
        """
        Obtiene los géneros más populares basados en el dataset actual
        """
        if self.anime_data is None:
            print("No hay datos de anime disponibles. Ejecute fetch_top_anime primero.")
            return []
            
        all_genres = []
        for genres in self.anime_data['genres']:
            all_genres.extend(genres)
            
        genre_counts = pd.Series(all_genres).value_counts()
        return genre_counts.to_dict()

# Función de utilidad para respetar el límite de tasa de la API
def respect_rate_limit():
    """Espera dos segundos para respetar el límite de la API Jikan"""
    time.sleep(2)  # Aumentado a 2 segundos para evitar errores de límite de tasa