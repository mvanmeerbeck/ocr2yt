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
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeMetadataUpdater:
    def __init__(self, credentials_file='youtube-tokens.json'):
        """Initialise le client YouTube API avec les credentials"""
        self.credentials_file = credentials_file
        self.youtube = None
        self._load_credentials()
    
    def _load_credentials(self):
        """Charge les credentials YouTube depuis le fichier JSON"""
        try:
            with open(self.credentials_file, 'r') as f:
                token_data = json.load(f)
            
            credentials = Credentials(
                token=token_data['access_token'],
                refresh_token=token_data['refresh_token'],
                token_uri='https://oauth2.googleapis.com/token',
                client_id=None,  # Sera récupéré automatiquement
                client_secret=None,  # Sera récupéré automatiquement
                scopes=['https://www.googleapis.com/auth/youtube.upload']
            )
            
            self.youtube = build('youtube', 'v3', credentials=credentials)
            logger.info("Connexion à l'API YouTube réussie")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des credentials: {e}")
            raise
    
    def _get_character_display_name(self, character):
        """Convertit le nom de personnage en format d'affichage"""
        character_names = {
            'ken': 'Ken',
            'marisa': 'Marisa',
            'juri': 'Juri',
            'chunli': 'Chun-Li',
            'akuma': 'Akuma',
            'sagat': 'Sagat'
        }
        return character_names.get(character.lower(), character.title())
    
    def _get_rank_display_name(self, rank):
        """Convertit le rang en format d'affichage"""
        rank_names = {
            'master': 'Master',
            'gm': 'Grand Master'
        }
        return rank_names.get(rank.lower(), rank.title())
    
    def _get_flag_display_name(self, flag):
        """Convertit le code pays en nom de pays"""
        country_names = {
            'fr': 'France',
            'uk': 'United Kingdom', 
            'se': 'Sweden',
            'ru': 'Russia',
            'sf': 'Street Fighter'  # Cas spécial
        }
        return country_names.get(flag.lower(), flag.upper())
    
    def _generate_title(self, players_data):
        """Génère un titre basé sur les données des joueurs"""
        if len(players_data) >= 2:
            p1 = players_data[0]
            p2 = players_data[1]
            
            p1_char = self._get_character_display_name(p1['character'])
            p2_char = self._get_character_display_name(p2['character'])
            
            return f"{p1['name']} ({p1_char}) vs {p2['name']} ({p2_char}) - Street Fighter 6"
        else:
            return "Street Fighter 6 Match"
    
    def _generate_description(self, players_data, video_id):
        """Génère une description basée sur les données des joueurs"""
        description_parts = [
            "🥊 Street Fighter 6 High Level Match 🥊\n",
            "Players:"
        ]
        
        for i, player in enumerate(players_data, 1):
            char_name = self._get_character_display_name(player['character'])
            rank_name = self._get_rank_display_name(player['rank'])
            country_name = self._get_flag_display_name(player['flag'])
            
            description_parts.append(
                f"Player {i}: {player['name']} ({char_name}) - {rank_name} - {country_name}"
            )
        
        description_parts.extend([
            "",
            "🎮 Street Fighter 6 competitive gameplay",
            "🏆 High level matches and tournaments",
            "",
            f"Video ID: {video_id}",
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(description_parts)
    
    def _generate_tags(self, players_data):
        """Génère des tags basés sur les données des joueurs"""
        base_tags = [
            "Street Fighter 6",
            "SF6",
            "Fighting Game",
            "FGC",
            "Tournament",
            "High Level",
            "Competitive"
        ]
        
        # Ajouter les personnages
        for player in players_data:
            char_name = self._get_character_display_name(player['character'])
            if char_name not in base_tags:
                base_tags.append(char_name)
        
        # Ajouter les rangs
        for player in players_data:
            rank_name = self._get_rank_display_name(player['rank'])
            if rank_name not in base_tags:
                base_tags.append(rank_name)
        
        return base_tags[:15]  # YouTube limite à 15 tags
    
    def update_video_metadata(self, video_id, players_data):
        """Met à jour les métadonnées d'une vidéo YouTube"""
        try:
            logger.info(f"Mise à jour de la vidéo {video_id}...")
            
            # Générer les métadonnées
            title = self._generate_title(players_data)
            description = self._generate_description(players_data, video_id)
            tags = self._generate_tags(players_data)
            
            # Construire la requête de mise à jour
            video_update = {
                'id': video_id,
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags,
                    'categoryId': '20'  # Gaming category
                }
            }
            
            # Exécuter la mise à jour
            self.youtube.videos().update(
                part='snippet',
                body=video_update
            ).execute()
            
            logger.info(f"✅ Vidéo {video_id} mise à jour avec succès")
            logger.info(f"   Titre: {title}")
            logger.info(f"   Tags: {', '.join(tags)}")
            
            return True
            
        except HttpError as e:
            logger.error(f"❌ Erreur HTTP lors de la mise à jour de {video_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour de {video_id}: {e}")
            return False
    
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
                if self.update_video_metadata(video_id, data['players']):
                    success_count += 1
                    
                    # Optionnel: déplacer le fichier traité vers un dossier 'done'
                    done_directory = os.path.join(os.path.dirname(todo_directory), 'done')
                    if not os.path.exists(done_directory):
                        os.makedirs(done_directory)
                    
                    done_path = os.path.join(done_directory, json_file)
                    os.rename(file_path, done_path)
                    logger.info(f"📁 Fichier déplacé vers: {done_path}")
                    
                else:
                    error_count += 1
                
                # Pause pour éviter de dépasser les limites de l'API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement de {json_file}: {e}")
                error_count += 1
        
        logger.info("\n📊 Résumé du traitement:")
        logger.info(f"   ✅ Succès: {success_count}")
        logger.info(f"   ❌ Erreurs: {error_count}")
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
