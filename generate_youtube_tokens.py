#!/usr/bin/env python3
"""
Script pour générer les tokens YouTube avec les bons scopes
"""

import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Scopes nécessaires pour modifier les métadonnées YouTube
SCOPES = [
    'https://www.googleapis.com/auth/youtube'
]

def generate_tokens():
    """Génère de nouveaux tokens YouTube avec les bons scopes"""
    
    # Récupérer les credentials depuis les variables d'environnement
    client_id = os.getenv('YOUTUBE_CLIENT_ID')
    client_secret = os.getenv('YOUTUBE_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("❌ YOUTUBE_CLIENT_ID et YOUTUBE_CLIENT_SECRET doivent être définis dans .env")
        return
    
    # Créer le fichier client_secrets.json temporaire
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": ["http://localhost"]
        }
    }
    
    print("🔐 Démarrage du processus d'authentification...")
    print(f"📋 Scopes demandés: {SCOPES}")
    
    # Créer le flow d'authentification
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    
    # Obtenir les credentials
    credentials = flow.run_local_server(port=0)
    
    # Sauvegarder les tokens
    token_data = {
        'access_token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'scope': ' '.join(credentials.scopes),
        'token_type': 'Bearer',
        'expiry_date': credentials.expiry.isoformat() if credentials.expiry else None
    }
    
    with open('youtube-tokens.json', 'w') as f:
        json.dump(token_data, f, indent=2)
    
    print("✅ Tokens générés et sauvegardés dans youtube-tokens.json")
    print(f"📋 Scopes obtenus: {credentials.scopes}")
    print("🎯 Vous pouvez maintenant utiliser metadata.py pour modifier les vidéos")

if __name__ == "__main__":
    generate_tokens()