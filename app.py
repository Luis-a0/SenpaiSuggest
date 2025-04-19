from flask import Flask, render_template, request, jsonify, redirect, url_for
from anime_recommender import AnimeRecommender, respect_rate_limit
import time

app = Flask(__name__)
recommender = AnimeRecommender()

# Cargar datos iniciales
# En versiones modernas de Flask, before_first_request está obsoleto
# Usamos esta alternativa
@app.route('/initialize')
def initialize_app():
    global recommender
    print("Cargando datos iniciales...")
    # Reducimos el límite para evitar errores con la API
    recommender.fetch_top_anime(limit=25)
    if recommender.anime_data is not None and not recommender.anime_data.empty:
        recommender.prepare_genre_matrix()
        print("Datos iniciales cargados correctamente.")
    else:
        print("No se pudieron cargar los datos iniciales.")
    return redirect(url_for('index'))

# Verificar datos antes de cada solicitud
@app.before_request
def load_data_if_needed():
    global recommender
    if recommender.anime_data is None or recommender.anime_data.empty:
        try:
            print("Intentando cargar datos iniciales...")
            recommender.fetch_top_anime(limit=25)
            if recommender.anime_data is not None and not recommender.anime_data.empty:
                recommender.prepare_genre_matrix()
                print("Datos iniciales cargados correctamente.")
        except Exception as e:
            print(f"Error al cargar datos: {e}")

# Cargar datos al inicio
with app.app_context():
    try:
        recommender.fetch_top_anime(limit=25)
        if recommender.anime_data is not None and not recommender.anime_data.empty:
            recommender.prepare_genre_matrix()
            print("Datos iniciales cargados al arranque.")
        else:
            print("No se pudieron cargar los datos iniciales al arranque.")
    except Exception as e:
        print(f"Error al cargar datos iniciales: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        return redirect(url_for('search_results', query=query))
    
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('index'))
    
    respect_rate_limit()
    results = recommender.search_anime(query)
    return render_template('results.html', query=query, results=results, type='search')

@app.route('/search_results')
def search_results():
    query = request.args.get('query', '')
    if not query:
        return redirect(url_for('index'))
    
    respect_rate_limit()
    results = recommender.search_anime(query)
    return render_template('results.html', query=query, results=results, type='search')

@app.route('/anime/<int:anime_id>')
def anime_details(anime_id):
    respect_rate_limit()
    anime = recommender.get_anime_details(anime_id)
    
    if not anime:
        return render_template('error.html', message="Anime no encontrado")
    
    respect_rate_limit()
    recommendations = recommender.get_recommendations(anime_id=anime_id)
    
    return render_template('anime.html', anime=anime, recommendations=recommendations)

@app.route('/recommend')
def recommend():
    genres = request.args.getlist('genres')
    respect_rate_limit()
    recommendations = recommender.recommend_by_genres(genres)
    
    return render_template('results.html', 
                          query=f"Géneros: {', '.join(genres)}", 
                          results=recommendations, 
                          type='recommendation')

@app.route('/popular_genres')
def popular_genres():
    genres = recommender.get_popular_genres()
    return jsonify(genres)

@app.route('/year/<int:year>')
def anime_by_year(year):
    respect_rate_limit()
    animes = recommender.get_anime_by_year(year)
    
    return render_template('results.html', 
                          query=f"Animes del año {year}", 
                          results=animes, 
                          type='year')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Página no encontrada"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Error del servidor"), 500

if __name__ == '__main__':
    app.run(debug=True)