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
            'ch': '🇨🇭'
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
        
        # Assembler la description finale
        description_parts = [
            main_description,
            "",
            " ".join(hashtags)
        ]
        
        return "\n".join(description_parts)
    
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
