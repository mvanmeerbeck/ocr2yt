#!/usr/bin/env python3
"""
Script pour mettre à jour les métadonnées des vidéos YouTube
basées sur les fichiers JSON dans ./metadata/todo
"""

import os
import json
import logging
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Variables d'environnement chargées depuis .env")
except ImportError:
    logger.warning("python-dotenv non installé, utilisation des variables d'environnement système")

class YouTubeMetadataUpdater:
    def __init__(self, credentials_file='youtube-tokens.json'):
        """Initialise le client YouTube API avec les credentials"""
        self.credentials_file = credentials_file
        self.youtube = None
        self.playlist_cache = {}  # Cache pour éviter de rechercher les mêmes playlists
        self._load_credentials()
    
    def _load_credentials(self):
        """Charge les credentials YouTube depuis le fichier JSON et les variables d'environnement"""
        try:
            # Charger les tokens depuis le fichier JSON
            with open(self.credentials_file, 'r') as f:
                token_data = json.load(f)
            
            # Récupérer client_id et client_secret depuis les variables d'environnement
            client_id = os.getenv('YOUTUBE_CLIENT_ID')
            client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
            
            # Vérifier que tous les champs requis sont présents
            if not token_data.get('access_token'):
                raise ValueError("access_token manquant dans le fichier tokens")
            if not token_data.get('refresh_token'):
                raise ValueError("refresh_token manquant dans le fichier tokens")
            if not client_id:
                raise ValueError("YOUTUBE_CLIENT_ID manquant dans les variables d'environnement")
            if not client_secret:
                raise ValueError("YOUTUBE_CLIENT_SECRET manquant dans les variables d'environnement")
            
            credentials = Credentials(
                token=token_data['access_token'],
                refresh_token=token_data['refresh_token'],
                token_uri='https://oauth2.googleapis.com/token',
                client_id=client_id,
                client_secret=client_secret,
                scopes=['https://www.googleapis.com/auth/youtube']
            )
            
            # Si les credentials sont expirés, ils seront automatiquement rafraîchis
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                logger.info("Token d'accès rafraîchi automatiquement")
                
                # Sauvegarder le nouveau token
                self._save_refreshed_token(credentials, token_data)
            
            self.youtube = build('youtube', 'v3', credentials=credentials)
            logger.info("Connexion à l'API YouTube réussie")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des credentials: {e}")
            raise
    
    def _save_refreshed_token(self, credentials, original_data):
        """Sauvegarde le token rafraîchi dans le fichier"""
        try:
            updated_data = original_data.copy()
            updated_data['access_token'] = credentials.token
            
            with open(self.credentials_file, 'w') as f:
                json.dump(updated_data, f, indent=2)
            
            logger.info(f"Token rafraîchi sauvegardé dans {self.credentials_file}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder le token rafraîchi: {e}")
    
    def _search_playlist(self, playlist_title):
        """Recherche une playlist par son titre"""
        try:
            # Vérifier le cache d'abord
            if playlist_title in self.playlist_cache:
                return self.playlist_cache[playlist_title]
            
            # Rechercher dans les playlists du channel
            request = self.youtube.playlists().list(
                part='snippet',
                mine=True,
                maxResults=50
            )
            
            while request:
                response = request.execute()
                
                for playlist in response['items']:
                    title = playlist['snippet']['title']
                    playlist_id = playlist['id']
                    
                    # Mettre en cache
                    self.playlist_cache[title] = playlist_id
                    
                    # Vérifier si c'est la playlist recherchée
                    if title.lower() == playlist_title.lower():
                        logger.info(f"📋 Playlist trouvée: '{playlist_title}' (ID: {playlist_id})")
                        return playlist_id
                
                # Page suivante
                request = self.youtube.playlists().list_next(request, response)
            
            logger.info(f"📋 Playlist '{playlist_title}' non trouvée")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche de playlist '{playlist_title}': {e}")
            return None
    
    def _create_playlist(self, title, description=""):
        """Crée une nouvelle playlist"""
        try:
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'defaultLanguage': 'fr'
                },
                'status': {
                    'privacyStatus': 'public'
                }
            }
            
            response = self.youtube.playlists().insert(
                part='snippet,status',
                body=body
            ).execute()
            
            playlist_id = response['id']
            
            # Mettre en cache
            self.playlist_cache[title] = playlist_id
            
            logger.info(f"✅ Playlist créée: '{title}' (ID: {playlist_id})")
            return playlist_id
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de playlist '{title}': {e}")
            return None
    
    def _get_or_create_playlist(self, title, description=""):
        """Récupère une playlist ou la crée si elle n'existe pas"""
        playlist_id = self._search_playlist(title)
        
        if playlist_id:
            return playlist_id
        else:
            logger.info(f"🔄 Création de la playlist '{title}'...")
            return self._create_playlist(title, description)
    
    def _add_video_to_playlist(self, video_id, playlist_id):
        """Ajoute une vidéo à une playlist"""
        try:
            body = {
                'snippet': {
                    'playlistId': playlist_id,
                    'resourceId': {
                        'kind': 'youtube#video',
                        'videoId': video_id
                    }
                }
            }
            
            self.youtube.playlistItems().insert(
                part='snippet',
                body=body
            ).execute()
            
            logger.info(f"📋 Vidéo {video_id} ajoutée à la playlist {playlist_id}")
            return True
            
        except HttpError as e:
            if 'videoAlreadyInPlaylist' in str(e):
                logger.info(f"📋 Vidéo {video_id} déjà dans la playlist")
                return True
            else:
                logger.error(f"❌ Erreur lors de l'ajout à la playlist: {e}")
                return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'ajout à la playlist: {e}")
            return False
    
    def _get_character_display_name(self, character):
        """Convertit le nom de personnage en format d'affichage"""
        character_names = {
            'ryu': 'Ryu',
            'chunli': 'Chun-Li',
            'ken': 'Ken',
            'zangief': 'Zangief',
            'dhalsim': 'Dhalsim',
            'blanka': 'Blanka',
            'ehonda': 'E.Honda',
            'guile': 'Guile',
            'cammy': 'Cammy',
            'deejay': 'Dee Jay',
            'juri': 'Juri',
            'akuma': 'Akuma',
            'luke': 'Luke',
            'jamie': 'Jamie',
            'manon': 'Manon',
            'marisa': 'Marisa',
            'jp': 'JP',
            'kimberly': 'Kimberly',
            'lily': 'Lily',
            'rashid': 'Rashid',
            'aki': 'A.K.I.',
            'ed': 'Ed',
            'terry': 'Terry',
            'mai': 'Mai',
            'elena': 'Elena',
            'bison': 'M.Bison',
        }
        return character_names.get(character.lower(), character.title())
    
    def _get_rank_display_name(self, rank):
        """Convertit le rang en format d'affichage"""
        rank_names = {
            'master': 'Master',
            'hm': 'High Master',
            'gm': 'Grand Master',
            'um': 'Ultimate Master',
            'legend': 'Legend',
        }
        return rank_names.get(rank.lower(), rank.title())
    
    def _get_flag_display_name(self, flag):
        """Convertit le code pays en nom de pays"""
        country_names = {
            'fr': '🇫🇷',
            'es': '🇪🇸',
            'uk': '🇬🇧', 
            'se': '🇸🇪',
            'ru': '🇷🇺',
            'sf': '🇺🇳',
            'kr': '🇰🇷',
            'de': '🇩🇪',
            'pl': '🇵🇱',
            'ch': '🇨🇭',
            'jm': '🇯🇲',
            'dk': '🇩🇰',
            'it': '🇮🇹',
            'be': '🇧🇪',
            'no': '🇳🇴',
            'fi': '🇫🇮',
            'lc': '🇱🇨',
            've': '🇻🇪',
        }
        return country_names.get(flag.lower(), flag.upper())
    
    def _generate_title(self, players_data):
        """Génère un titre basé sur les données des joueurs"""
        p1 = players_data[0]
        p2 = players_data[1]
        
        p1_char = self._get_character_display_name(p1['character'])
        p2_char = self._get_character_display_name(p2['character'])

        return f"{self._get_flag_display_name(p1['flag'])} {p1['name']} ({p1_char}) vs {self._get_flag_display_name(p2['flag'])} {p2['name']} ({p2_char}) | Street Fighter 6"

    def _generate_description(self, players_data, video_id):
        """Génère une description basée sur les données des joueurs"""
        p1 = players_data[0]
        p2 = players_data[1]    
        
        # Obtenir les noms d'affichage
        p1_char = self._get_character_display_name(p1['character'])
        p2_char = self._get_character_display_name(p2['character'])
        p1_flag = self._get_flag_display_name(p1['flag'])
        p2_flag = self._get_flag_display_name(p2['flag'])
        
        # Description principale selon le pattern demandé
        main_description = f"Matchs de Street Fighter 6 entre {p1_flag} {p1['name']} ({p1_char}) et {p2_flag} {p2['name']} ({p2_char})."
        
        # Générer les hashtags
        hashtags = ["#streetfighter", "#streetfighter6", "#sf6"]
        
        # Ajouter les hashtags pour les personnages
        for player in [p1, p2]:
            char = player['character'].lower()
            char_hashtag = f"#sf6{char}"
            if char_hashtag not in hashtags:
                hashtags.append(char_hashtag)
        
        # Ajouter hashtag Modern si détecté
        modern_detected = False
        for player in [p1, p2]:
            name = player.get('name', '').lower()
            if 'modern' in name or player.get('control_type') == 'modern':
                modern_detected = True
                break
        
        if modern_detected and "#sf6modern" not in hashtags:
            hashtags.append("#sf6modern")
        
        # Ajouter hashtags des rangs
        ranks = []
        for player in [p1, p2]:
            rank = player.get('rank', '').lower()
            if rank:
                ranks.append(rank)
        
        if ranks:
            # Hiérarchie des rangs pour prendre le plus élevé
            rank_hierarchy = ['legend', 'um', 'gm', 'hm', 'master']
            
            highest_rank = None
            for rank in rank_hierarchy:
                if rank in ranks:
                    highest_rank = rank
                    break
            
            if highest_rank:
                # Convertir en hashtag
                if highest_rank == 'um':
                    rank_hashtag = "#sf6ultimatemaster"
                elif highest_rank == 'gm':
                    rank_hashtag = "#sf6grandmaster"
                elif highest_rank == 'hm':
                    rank_hashtag = "#sf6highmaster"
                else:
                    rank_hashtag = f"#sf6{highest_rank}"
                
                if rank_hashtag not in hashtags:
                    hashtags.append(rank_hashtag)
        
        # Assembler la description finale
        description_parts = [
            main_description,
            "",
            " ".join(hashtags)
        ]
        
        return "\n".join(description_parts)
    
    def _manage_playlists(self, video_id, players_data):
        """Gère l'ajout de la vidéo aux playlists appropriées"""
        playlists_added = []
        
        try:
            # Récupérer les informations des joueurs
            p1 = players_data[0]
            p2 = players_data[1]
            
            # 1. Playlists par personnage
            for player in [p1, p2]:
                character = player.get('character', '').lower()
                if character:
                    char_display = self._get_character_display_name(character)
                    playlist_title = f"{char_display} | Street Fighter 6"
                    playlist_description = f"Toutes les vidéos avec {char_display} dans Street Fighter 6"
                    
                    playlist_id = self._get_or_create_playlist(playlist_title, playlist_description)
                    if playlist_id and self._add_video_to_playlist(video_id, playlist_id):
                        playlists_added.append(playlist_title)
            
            # 2. Playlist Modern (si au moins un joueur utilise Modern)
            # On peut détecter Modern si le nom contient certains indicateurs
            # Ou ajouter cette info dans les données JSON
            modern_detected = False
            for player in [p1, p2]:
                name = player.get('name', '').lower()
                # Vous pouvez ajuster ces critères selon vos besoins
                if 'modern' in name or player.get('control_type') == 'modern':
                    modern_detected = True
                    break
            
            if modern_detected:
                playlist_title = "Modern | Street Fighter 6"
                playlist_description = "Matchs Street Fighter 6 avec contrôles Modern"
                playlist_id = self._get_or_create_playlist(playlist_title, playlist_description)
                if playlist_id and self._add_video_to_playlist(video_id, playlist_id):
                    playlists_added.append(playlist_title)
            
            # 3. Playlists par rang (basé sur le rang le plus élevé)
            ranks = []
            for player in [p1, p2]:
                rank = player.get('rank', '').lower()
                if rank:
                    ranks.append(rank)
            
            if ranks:
                # Hiérarchie des rangs (du plus élevé au plus bas)
                rank_hierarchy = ['legend', 'um', 'gm', 'master']
                
                highest_rank = None
                for rank in rank_hierarchy:
                    if rank in ranks:
                        highest_rank = rank
                        break
                
                if highest_rank:
                    rank_display = self._get_rank_display_name(highest_rank)
                    playlist_title = f"{rank_display} | Street Fighter 6"
                    playlist_description = f"Matchs Street Fighter 6 de niveau {rank_display}"
                    playlist_id = self._get_or_create_playlist(playlist_title, playlist_description)
                    if playlist_id and self._add_video_to_playlist(video_id, playlist_id):
                        playlists_added.append(playlist_title)
            
            if playlists_added:
                logger.info(f"📋 Vidéo ajoutée aux playlists: {', '.join(playlists_added)}")
            
            return playlists_added
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la gestion des playlists pour {video_id}: {e}")
            return []
    
    def update_video_metadata(self, video_id, players_data):
        """Met à jour les métadonnées d'une vidéo YouTube
        
        Returns:
            str: 'success', 'forbidden', 'error'
        """
        try:
            logger.info(f"Mise à jour de la vidéo {video_id}...")
            
            # Générer les métadonnées
            title = self._generate_title(players_data)
            description = self._generate_description(players_data, video_id)
            
            # Construire la requête de mise à jour
            video_update = {
                'id': video_id,
                'snippet': {
                    'title': title,
                    'description': description,
                    'categoryId': '20',
                    'defaultLanguage': 'fr',
                    'defaultAudioLanguage': 'fr'
                }
            }

            self.youtube.videos().update(
                part='snippet',
                body=video_update
            ).execute()
            
            logger.info(video_update)
            
            # Gérer les playlists après la mise à jour réussie
            self._manage_playlists(video_id, players_data)
            
            return 'success'
            
        except HttpError as e:
            if e.resp.status == 403:
                if 'forbidden' in str(e).lower():
                    logger.warning(f"🚫 Vidéo {video_id}: Accès interdit (peut-être pas votre vidéo ou restrictions)")
                elif 'insufficient' in str(e).lower():
                    logger.warning(f"🔐 Vidéo {video_id}: Permissions insuffisantes")
                else:
                    logger.warning(f"🚫 Vidéo {video_id}: Erreur 403 - {e}")
                return 'forbidden'
            else:
                logger.error(f"❌ Erreur HTTP lors de la mise à jour de {video_id}: {e}")
                return 'error'
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour de {video_id}: {e}")
            return 'error'
    
    def process_metadata_files(self, todo_directory='./metadata/todo'):
        """Traite tous les fichiers JSON dans le dossier todo"""
        if not os.path.exists(todo_directory):
            logger.error(f"Le dossier {todo_directory} n'existe pas")
            return
        
        json_files = [f for f in os.listdir(todo_directory) if f.endswith('.json')]
        
        if not json_files:
            logger.info("Aucun fichier JSON trouvé dans le dossier todo")
            return
        
        logger.info(f"Traitement de {len(json_files)} fichier(s) JSON...")
        
        success_count = 0
        error_count = 0
        forbidden_count = 0
        
        for json_file in json_files:
            try:
                # Extraire l'ID de la vidéo depuis le nom du fichier
                video_id = os.path.splitext(json_file)[0]
                file_path = os.path.join(todo_directory, json_file)
                
                # Charger les données JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'players' not in data:
                    logger.warning(f"❌ Fichier {json_file}: pas de données 'players'")
                    error_count += 1
                    continue
                
                # Mettre à jour la vidéo
                result = self.update_video_metadata(video_id, data['players'])
                
                if result == 'success':
                    success_count += 1
                    
                    # Optionnel: déplacer le fichier traité vers un dossier 'done'
                    done_directory = os.path.join(os.path.dirname(todo_directory), 'done')
                    if not os.path.exists(done_directory):
                        os.makedirs(done_directory)
                    
                    done_path = os.path.join(done_directory, json_file)
                    # os.rename(file_path, done_path)
                    logger.info(f"📁 Fichier déplacé vers: {done_path}")
                    
                elif result == 'forbidden':
                    forbidden_count += 1
                else:
                    error_count += 1
                
                # Pause pour éviter de dépasser les limites de l'API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {json_file}: {e}")
                error_count += 1
        
        logger.info("\n📊 Résumé du traitement:")
        logger.info(f"   ✅ Succès: {success_count}")
        logger.info(f"   🚫 Accès interdit (403): {forbidden_count}")
        logger.info(f"   ❌ Autres erreurs: {error_count}")
        logger.info(f"   📁 Total: {len(json_files)}")


def main():
    """Fonction principale"""
    try:
        updater = YouTubeMetadataUpdater()
        updater.process_metadata_files()
    except KeyboardInterrupt:
        logger.info("❌ Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")


if __name__ == "__main__":
    main()
