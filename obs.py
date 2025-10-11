#!/usr/bin/env python3
"""
Script pour se connecter à OBS via WebSocket
"""

import os
import json
import logging
import asyncio
import websockets
import subprocess
from pathlib import Path
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OBSWebSocketClient:
    def __init__(self):
        """Initialise le client WebSocket OBS"""
        self.websocket_url = os.getenv('OBS_WEBSOCKET_URL', 'ws://localhost:4455')
        self.websocket_password = os.getenv('OBS_WEBSOCKET_PASSWORD', '')
        self.websocket = None
        self.is_connected = False
        
    async def connect(self):
        """Se connecte au WebSocket OBS"""
        try:
            logger.info(f"Connexion à OBS WebSocket: {self.websocket_url}")
            
            self.websocket = await websockets.connect(self.websocket_url)
            self.is_connected = True
            
            logger.info("✅ Connexion WebSocket OBS établie")
            
            # Écouter les messages entrants
            await self._listen_messages()
            
        except Exception as e:
            logger.error(f"❌ Erreur de connexion WebSocket OBS: {e}")
            self.is_connected = False
    
    async def _listen_messages(self):
        """Écoute les messages entrants du WebSocket"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self._handle_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("🔌 Connexion WebSocket fermée")
            self.is_connected = False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'écoute des messages: {e}")
            self.is_connected = False
    
    async def _handle_message(self, data):
        """Traite les messages reçus d'OBS"""
        # Logger le message JSON complet
        logger.info("📨 Message reçu d'OBS:")
        logger.info(f"   Raw JSON: {json.dumps(data, indent=2)}")
        
        message_type = data.get('op')  # Operation code
        message_data = data.get('d', {})
        
        logger.info(f"   Type de message (op): {message_type}")
        
        if message_type == 0:  # Hello message
            logger.info("👋 Message Hello reçu d'OBS")
            logger.info(f"   Données Hello: {json.dumps(message_data, indent=2)}")
            await self._authenticate(data)
        elif message_type == 2:  # Identify response
            logger.info("🔐 Identification réussie")
            logger.info(f"   Données d'identification: {json.dumps(message_data, indent=2)}")
        elif message_type == 5:  # Event
            event_type = message_data.get('eventType')
            event_data = message_data.get('eventData', {})
            logger.info(f"📺 Événement OBS: {event_type}")
            logger.info(f"   Données de l'événement: {json.dumps(event_data, indent=2)}")
            await self._handle_event(data)
        elif message_type == 7:  # Request response
            request_type = message_data.get('requestType')
            request_status = message_data.get('requestStatus', {})
            logger.info(f"📤 Réponse à la requête: {request_type}")
            logger.info(f"   Statut: {json.dumps(request_status, indent=2)}")
            await self._handle_request_response(data)
        else:
            logger.info(f"❓ Type de message inconnu: {message_type}")
            logger.info(f"   Données complètes: {json.dumps(data, indent=2)}")
    
    async def _authenticate(self, hello_data):
        """Authentification avec OBS WebSocket"""
        try:
            # Message d'identification
            identify_message = {
                "op": 1,  # Identify
                "d": {
                    "rpcVersion": 1
                }
            }
            
            # Ajouter le mot de passe si fourni
            if self.websocket_password:
                # TODO: Implémenter l'authentification avec hash si nécessaire
                identify_message["d"]["authentication"] = self.websocket_password
            
            await self.websocket.send(json.dumps(identify_message))
            logger.info("🔐 Message d'identification envoyé")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'authentification: {e}")
    
    async def _handle_event(self, data):
        """Traite les événements OBS"""
        event_type = data.get('d', {}).get('eventType')
        event_data = data.get('d', {}).get('eventData', {})
        
        logger.info(f"📺 Événement OBS: {event_type}")
        
        if event_type == 'RecordStateChanged':
            output_state = event_data.get('outputState')
            output_path = event_data.get('outputPath')
            
            # État d'enregistrement selon le nouveau format OBS WebSocket v5
            if output_state == 'OBS_WEBSOCKET_OUTPUT_STARTED':
                logger.info("🔴 Enregistrement démarré")
            elif output_state == 'OBS_WEBSOCKET_OUTPUT_STOPPED':
                logger.info("⏹️ Enregistrement arrêté")
                
                # 🎯 Auto-upload quand l'enregistrement s'arrête
                if output_path:
                    logger.info(f"📹 Recording stopped: {output_path}")
                    await self._handle_recording_stopped(output_path)
                else:
                    logger.warning("⚠️ Pas de chemin de fichier fourni pour l'enregistrement arrêté")
            else:
                # Fallback pour l'ancien format (outputActive)
                recording_active = event_data.get('outputActive', False)
                if recording_active:
                    logger.info("🔴 Enregistrement démarré (format legacy)")
                else:
                    logger.info("⏹️ Enregistrement arrêté (format legacy)")
                    if output_path:
                        await self._handle_recording_stopped(output_path)
        
        elif event_type == 'StreamStateChanged':
            streaming_active = event_data.get('outputActive', False)
            if streaming_active:
                logger.info("📡 Stream démarré")
            else:
                logger.info("📡 Stream arrêté")
    
    async def _handle_recording_stopped(self, output_path):
        """Gère l'arrêt d'enregistrement et prépare l'upload YouTube"""
        try:
            logger.info(f"🎬 Traitement de l'enregistrement arrêté: {output_path}")
            
            # Vérifier que le fichier existe
            if not os.path.exists(output_path):
                logger.error(f"❌ Fichier vidéo introuvable: {output_path}")
                return
            
            # Extraire la première frame
            thumbnail_path = await self._extract_first_frame(output_path)
            
            # Vérifier l'authentification YouTube
            has_youtube_auth = await self._check_youtube_auth()
            
            if has_youtube_auth:
                logger.info("✅ Authentification YouTube disponible")
                
                # Upload automatique sur YouTube
                try:
                    video_name = os.path.splitext(os.path.basename(output_path))[0]
                    title = f"Street Fighter 6 - {video_name}"
                    description = "Match Street Fighter 6 enregistré automatiquement depuis OBS"
                    
                    result = await self.upload_to_youtube(
                        video_path=output_path,
                        title=title,
                        description=description,
                        thumbnail_path=thumbnail_path,
                        delete_after_upload=False  # Garder le fichier par défaut
                    )
                    
                    logger.info(f"🎉 Upload YouTube terminé: {result['id']}")
                    
                except Exception as upload_error:
                    logger.error(f"❌ Erreur durant l'upload YouTube: {upload_error}")
            else:
                logger.warning("❌ Pas d'authentification YouTube - Exécutez la configuration YouTube")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de l'arrêt d'enregistrement: {e}")
    
    async def _extract_first_frame(self, video_path, output_path=None):
        """Extrait la première frame d'une vidéo avec FFmpeg"""
        try:
            logger.info(f"🎥 Extraction de la première frame depuis: {video_path}")
            
            # Vérifier que le fichier vidéo existe
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Fichier vidéo introuvable: {video_path}")
            
            # Créer le chemin de sortie si non spécifié
            if not output_path:
                video_dir = os.path.dirname(video_path)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(video_dir, f"{video_name}_thumbnail.jpg")
            
            logger.info(f"📁 Chemin de sortie: {output_path}")
            
            # Commande FFmpeg pour extraire la frame avec compression JPEG
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vframes', '1',
                '-q:v', '1',
                '-y',  # Overwrite output file
                output_path
            ]
            
            logger.info(f"🔧 Commande FFmpeg: {' '.join(ffmpeg_cmd)}")
            
            # Exécuter FFmpeg
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Vérifier que le fichier a été créé
                if os.path.exists(output_path):
                    logger.info(f"✅ Première frame extraite vers: {output_path}")
                    return output_path
                else:
                    logger.error("❌ Le fichier de sortie n'a pas été créé")
                    return None
            else:
                stderr_text = stderr.decode('utf-8') if stderr else "Pas d'erreur stderr"
                logger.error(f"❌ FFmpeg a échoué (code {process.returncode}): {stderr_text}")
                return None
                
        except FileNotFoundError as e:
            if "ffmpeg" in str(e):
                logger.error("❌ FFmpeg n'est pas installé ou pas dans le PATH")
                logger.info("💡 Installez FFmpeg: https://ffmpeg.org/download.html")
            else:
                logger.error(f"❌ Fichier introuvable: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction de la frame: {e}")
            return None
    
    async def _check_youtube_auth(self):
        """Vérifie si l'authentification YouTube est disponible"""
        try:
            # Vérifier les variables d'environnement YouTube
            client_id = os.getenv('YOUTUBE_CLIENT_ID')
            client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
            
            # Vérifier le fichier de tokens
            tokens_file = 'youtube-tokens.json'
            tokens_exist = os.path.exists(tokens_file)
            
            if client_id and client_secret and tokens_exist:
                logger.info("🔐 Credentials YouTube trouvés")
                return True
            else:
                missing = []
                if not client_id:
                    missing.append("YOUTUBE_CLIENT_ID")
                if not client_secret:
                    missing.append("YOUTUBE_CLIENT_SECRET") 
                if not tokens_exist:
                    missing.append("youtube-tokens.json")
                
                logger.warning(f"❌ Éléments manquants pour YouTube: {', '.join(missing)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de l'auth YouTube: {e}")
            return False
    
    def _get_youtube_service(self):
        """Crée le service YouTube API"""
        try:
            # Charger les tokens
            with open('youtube-tokens.json', 'r') as f:
                token_data = json.load(f)
            
            # Récupérer les credentials depuis l'environnement
            client_id = os.getenv('YOUTUBE_CLIENT_ID')
            client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
            
            credentials = Credentials(
                token=token_data['access_token'],
                refresh_token=token_data['refresh_token'],
                token_uri='https://oauth2.googleapis.com/token',
                client_id=client_id,
                client_secret=client_secret,
                scopes=['https://www.googleapis.com/auth/youtube']
            )
            
            # Rafraîchir si nécessaire
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            
            return build('youtube', 'v3', credentials=credentials)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du service YouTube: {e}")
            return None
    
    async def upload_to_youtube(self, video_path, title=None, description=None, thumbnail_path=None, delete_after_upload=False):
        """Upload une vidéo sur YouTube"""
        try:
            logger.info("🚀 Démarrage de l'upload YouTube...")
            
            # Vérifier que le fichier existe
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Fichier introuvable: {video_path}")
            
            # Créer le service YouTube
            youtube = self._get_youtube_service()
            if not youtube:
                raise Exception("Impossible de créer le service YouTube")
            
            # Paramètres par défaut
            if not title:
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                title = f"Street Fighter 6 - {video_name}"
            
            if not description:
                description = "Vidéo uploadée automatiquement depuis OBS"
            
            # Métadonnées de la vidéo
            video_metadata = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'categoryId': '20',  # Gaming
                    'defaultLanguage': 'fr',
                    'defaultAudioLanguage': 'fr'
                },
                'status': {
                    'privacyStatus': 'private'  # Privé par défaut
                }
            }
            
            # Informations sur le fichier
            file_size = os.path.getsize(video_path)
            logger.info(f"📁 Fichier: {os.path.basename(video_path)}")
            logger.info(f"📊 Taille: {file_size / (1024 * 1024):.2f} MB")
            
            # Créer le media upload
            media = MediaFileUpload(
                video_path,
                chunksize=-1,  # Upload en une seule fois
                resumable=True
            )
            
            # Upload avec progression
            request = youtube.videos().insert(
                part='snippet,status',
                body=video_metadata,
                media_body=media
            )
            
            # Exécuter l'upload
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.info(f"📤 Upload en cours: {progress}%")
            
            video_id = response['id']
            logger.info("✅ Upload terminé!")
            logger.info(f"🎬 Video ID: {video_id}")
            logger.info(f"🔗 URL: https://www.youtube.com/watch?v={video_id}")
            
            # Upload du thumbnail si disponible
            if thumbnail_path and os.path.exists(thumbnail_path):
                try:
                    new_thumbnail_path = await self._upload_thumbnail(youtube, video_id, thumbnail_path)
                    thumbnail_path = new_thumbnail_path  # Mettre à jour le chemin pour la suppression
                except Exception as thumb_error:
                    logger.warning(f"⚠️ Erreur upload thumbnail: {thumb_error}")
            
            # Supprimer le fichier si demandé
            if delete_after_upload:
                try:
                    os.remove(video_path)
                    logger.info(f"🗑️ Fichier supprimé: {video_path}")
                    
                    # Supprimer aussi le thumbnail
                    if thumbnail_path and os.path.exists(thumbnail_path):
                        os.remove(thumbnail_path)
                        logger.info(f"🗑️ Thumbnail supprimé: {thumbnail_path}")
                        
                except OSError as delete_error:
                    logger.warning(f"⚠️ Impossible de supprimer le fichier: {delete_error}")
            
            return response
            
        except HttpError as e:
            logger.error(f"❌ Erreur HTTP YouTube: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Erreur upload YouTube: {e}")
            raise
    
    async def _upload_thumbnail(self, youtube, video_id, thumbnail_path):
        """Upload un thumbnail pour la vidéo et le déplace"""
        try:
            logger.info(f"🖼️ Upload du thumbnail: {thumbnail_path}")
            
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path)
            ).execute()
            
            logger.info("✅ Thumbnail uploadé avec succès")
            
            # Déplacer le thumbnail vers le dossier thumbnails avec l'ID vidéo
            original_dir = os.path.dirname(thumbnail_path)
            original_ext = os.path.splitext(thumbnail_path)[1]
            
            # Créer le dossier thumbnails s'il n'existe pas
            thumbnails_dir = os.path.join(original_dir, 'thumbnails')
            os.makedirs(thumbnails_dir, exist_ok=True)
            
            # Nouveau chemin avec l'ID vidéo
            new_thumbnail_path = os.path.join(thumbnails_dir, f"{video_id}{original_ext}")
            
            logger.info(f"📝 Déplacement du thumbnail: {thumbnail_path} > {new_thumbnail_path}")
            
            # Déplacer le fichier
            os.rename(thumbnail_path, new_thumbnail_path)
            logger.info(f"✅ Thumbnail déplacé avec succès: {new_thumbnail_path}")
            
            return new_thumbnail_path
            
        except Exception as e:
            logger.error(f"❌ Erreur upload/déplacement thumbnail: {e}")
            raise
    
    async def _handle_request_response(self, data):
        """Traite les réponses aux requêtes"""
        request_type = data.get('d', {}).get('requestType')
        request_status = data.get('d', {}).get('requestStatus', {})
        
        if request_status.get('result'):
            logger.info(f"✅ Requête {request_type} réussie")
        else:
            logger.error(f"❌ Requête {request_type} échouée: {request_status.get('comment', 'Erreur inconnue')}")
    
    async def send_request(self, request_type, request_data=None):
        """Envoie une requête à OBS"""
        if not self.is_connected:
            logger.error("❌ Pas de connexion WebSocket active")
            return None
        
        try:
            message = {
                "op": 6,  # Request
                "d": {
                    "requestType": request_type,
                    "requestId": f"req_{request_type}_{asyncio.get_event_loop().time()}"
                }
            }
            
            if request_data:
                message["d"]["requestData"] = request_data
            
            await self.websocket.send(json.dumps(message))
            logger.info(f"📤 Requête envoyée: {request_type}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi de la requête: {e}")
    
    async def get_recording_status(self):
        """Récupère le statut d'enregistrement"""
        await self.send_request("GetRecordStatus")
    
    async def start_recording(self):
        """Démarre l'enregistrement"""
        await self.send_request("StartRecord")
    
    async def stop_recording(self):
        """Arrête l'enregistrement"""
        await self.send_request("StopRecord")
    
    async def disconnect(self):
        """Ferme la connexion WebSocket"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            logger.info("🔌 Connexion WebSocket fermée")

async def main():
    """Fonction principale pour tester la connexion"""
    client = OBSWebSocketClient()
    
    try:
        await client.connect()
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt demandé par l'utilisateur")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())