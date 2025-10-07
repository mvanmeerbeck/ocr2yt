#!/usr/bin/env python3
"""
Générateur de templates de caractères à partir de fonts.

Ce script génère des images de templates pour chaque caractère en utilisant
les fonts disponibles dans le dossier ./fonts/

Structure de sortie:
./assets/templates/names/a/a-font-1.png
./assets/templates/names/b/b-font-1.png
etc.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import string
import argparse


def create_character_template(char, font_path, size=40, output_size=(40, 40)):
    """
    Crée un template d'image pour un caractère donné.
    
    Args:
        char: Le caractère à générer
        font_path: Chemin vers le fichier de font
        size: Taille de la font
        output_size: Taille finale de l'image (width, height)
    
    Returns:
        numpy array de l'image générée
    """
    try:
        # Charger la font
        font = ImageFont.truetype(str(font_path), size)
    except Exception as e:
        print(f"❌ Erreur lors du chargement de la font {font_path}: {e}")
        return None
    
    # Créer une image temporaire pour mesurer la taille du texte
    temp_img = Image.new('RGB', (100, 100), color='white')
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Mesurer la taille du caractère
    bbox = temp_draw.textbbox((0, 0), char, font=font)
    char_width = bbox[2] - bbox[0]
    char_height = bbox[3] - bbox[1]
    
    # Créer l'image finale avec la taille spécifiée
    img = Image.new('RGB', output_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Centrer le caractère dans l'image
    x = (output_size[0] - char_width) // 2
    y = (output_size[1] - char_height) // 2
    
    # Dessiner le caractère en noir
    draw.text((x, y), char, font=font, fill='black')
    
    # Convertir en numpy array pour OpenCV
    img_array = np.array(img)
    
    # Convertir en niveaux de gris
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    return gray_img


def get_character_sets():
    """
    Retourne les différents ensembles de caractères à générer.
    """
    character_sets = {
        'letters_lower': string.ascii_lowercase,  # a-z
        'letters_upper': string.ascii_uppercase,  # A-Z
        'digits': string.digits,  # 0-9
        'punctuation': ['.', ',', ':', ';', '!', '?', '-', '_', '(', ')', '[', ']', '/', '\\'],
        'special': [' ']  # espace
    }
    
    return character_sets


def get_safe_filename(char):
    """
    Retourne un nom de fichier sûr pour un caractère donné.
    """
    char_mapping = {
        ' ': 'space',
        '.': 'dot',
        ',': 'comma',
        ':': 'colon',
        ';': 'semicolon',
        '!': 'exclamation',
        '?': 'question',
        '-': 'hyphen',
        '_': 'underscore',
        '(': 'lparen',
        ')': 'rparen',
        '[': 'lbracket',
        ']': 'rbracket',
        '/': 'slash',
        '\\': 'backslash',
    }
    
    return char_mapping.get(char, char)


def generate_templates_for_font(font_path, output_base_dir, template_size=40, categories=None):
    """
    Génère tous les templates pour une font donnée.
    
    Args:
        font_path: Chemin vers le fichier de font
        output_base_dir: Dossier de base pour les templates
        template_size: Taille du template (carré)
        categories: Liste des catégories à générer (None = toutes)
    """
    font_name = Path(font_path).stem
    character_sets = get_character_sets()
    
    if categories:
        character_sets = {k: v for k, v in character_sets.items() if k in categories}
    
    print(f"\n=== Génération des templates pour {font_name} ===")
    print(f"Font: {font_path}")
    print(f"Taille: {template_size}x{template_size}")
    print(f"Catégories: {list(character_sets.keys())}")
    
    total_generated = 0
    
    for category_name, chars in character_sets.items():
        print(f"\n--- Catégorie: {category_name} ---")
        
        for char in chars:
            # Créer le template
            template_img = create_character_template(
                char, font_path, size=template_size, output_size=(template_size, template_size)
            )
            
            if template_img is not None:
                # Créer le dossier pour ce caractère
                safe_char_name = get_safe_filename(char)
                char_dir = output_base_dir / "names" / safe_char_name
                char_dir.mkdir(parents=True, exist_ok=True)
                
                # Nom du fichier template
                template_filename = f"{safe_char_name}-{font_name}-1.png"
                template_path = char_dir / template_filename
                
                # Sauvegarder l'image
                cv2.imwrite(str(template_path), template_img)
                
                print(f"  ✅ '{char}' -> {template_path}")
                total_generated += 1
            else:
                print(f"  ❌ Échec pour '{char}'")
    
    print(f"\n✅ Total généré: {total_generated} templates")
    return total_generated


def generate_multiple_sizes(font_path, output_base_dir, sizes=[32, 36, 40, 44, 48]):
    """
    Génère des templates en plusieurs tailles pour une font.
    """
    font_name = Path(font_path).stem
    print(f"\n=== Génération multi-tailles pour {font_name} ===")
    
    total_generated = 0
    
    # Caractères de base (lettres + chiffres + ponctuation courante)
    basic_chars = (
        string.ascii_lowercase + 
        string.ascii_uppercase + 
        string.digits + 
        '.,:-_()[]'
    )
    
    for char in basic_chars:
        safe_char_name = get_safe_filename(char)
        char_dir = output_base_dir / "names" / safe_char_name
        char_dir.mkdir(parents=True, exist_ok=True)
        
        for i, size in enumerate(sizes, 1):
            template_img = create_character_template(
                char, font_path, size=size, output_size=(40, 40)  # Toujours 40x40 en sortie
            )
            
            if template_img is not None:
                template_filename = f"{safe_char_name}-{font_name}-{i}.png"
                template_path = char_dir / template_filename
                cv2.imwrite(str(template_path), template_img)
                total_generated += 1
        
        print(f"  ✅ '{char}' -> {len(sizes)} tailles générées")
    
    print(f"✅ Total multi-tailles: {total_generated} templates")
    return total_generated


def main():
    parser = argparse.ArgumentParser(description='Générateur de templates de caractères')
    parser.add_argument('--fonts-dir', type=str, default='./fonts', 
                       help='Dossier contenant les fonts (défaut: ./fonts)')
    parser.add_argument('--output-dir', type=str, default='./assets/templates',
                       help='Dossier de sortie (défaut: ./assets/templates)')
    parser.add_argument('--size', type=int, default=40,
                       help='Taille des templates en pixels (défaut: 40)')
    parser.add_argument('--multi-sizes', action='store_true',
                       help='Générer plusieurs tailles par caractère')
    parser.add_argument('--categories', nargs='+', 
                       choices=['letters_lower', 'letters_upper', 'digits', 'punctuation', 'special'],
                       help='Catégories à générer (défaut: toutes)')
    
    args = parser.parse_args()
    
    # Vérifications
    fonts_dir = Path(args.fonts_dir)
    output_dir = Path(args.output_dir)
    
    if not fonts_dir.exists():
        print(f"❌ Dossier fonts non trouvé: {fonts_dir}")
        return
    
    # Trouver tous les fichiers de fonts
    font_files = list(fonts_dir.glob('*.ttf')) + list(fonts_dir.glob('*.otf'))
    
    if not font_files:
        print(f"❌ Aucune font trouvée dans {fonts_dir}")
        print("   Extensions supportées: .ttf, .otf")
        return
    
    print(f"Fonts trouvées: {len(font_files)}")
    for font_file in font_files:
        print(f"  - {font_file.name}")
    
    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_templates = 0
    
    # Générer les templates pour chaque font
    for font_file in font_files:
        try:
            if args.multi_sizes:
                generated = generate_multiple_sizes(font_file, output_dir)
            else:
                generated = generate_templates_for_font(
                    font_file, output_dir, args.size, args.categories
                )
            total_templates += generated
        except Exception as e:
            print(f"❌ Erreur avec la font {font_file}: {e}")
    
    print("\n🎉 TERMINÉ!")
    print(f"Total de templates générés: {total_templates}")
    print(f"Templates sauvegardés dans: {output_dir}")
    print("\nStructure créée:")
    print(f"  {output_dir}/names/a/a-fontname-1.png")
    print(f"  {output_dir}/names/b/b-fontname-1.png")
    print("  etc.")


if __name__ == "__main__":
    main()
