#!/usr/bin/env python3
"""
Script pour découper les fichiers zone_preview en lettres individuelles
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
import pytesseract


def extract_letter_colors_from_samples(sample_images_dir):
    """
    Extrait automatiquement les couleurs de lettres à partir d'images d'exemple.
    
    Args:
        sample_images_dir: Dossier contenant les images d'exemple
    
    Returns:
        Liste des couleurs uniques trouvées
    """
    print(f"🔍 Analyse des couleurs dans {sample_images_dir}...")
    
    all_colors = set()
    sample_files = []
    
    # Chercher tous les fichiers images dans le dossier
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        sample_files.extend(glob.glob(os.path.join(sample_images_dir, ext)))
    
    if not sample_files:
        print(f"❌ Aucune image trouvée dans {sample_images_dir}")
        return []
    
    print(f"📁 Images d'exemple trouvées: {len(sample_files)}")
    
    for sample_file in sample_files:
        print(f"  📄 Analyse: {os.path.basename(sample_file)}")
        
        # Charger l'image
        img = cv2.imread(sample_file)
        if img is None:
            continue
            
        # Convertir BGR vers RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionner pour accélérer l'analyse si l'image est très grande
        height, width = img_rgb.shape[:2]
        if width > 500 or height > 500:
            scale = min(500/width, 500/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height))
        
        # Extraire toutes les couleurs uniques
        pixels = img_rgb.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        
        # Filtrer les couleurs trop sombres (fond) ou trop claires (blanc pur)
        for color in unique_colors:
            r, g, b = color
            
            # Éviter le noir pur, blanc pur et les couleurs très sombres
            if (r < 50 and g < 50 and b < 50) or (r > 250 and g > 250 and b > 250):
                continue
            
            # Éviter les couleurs de fond typiques (très saturées ou grises pures)
            if abs(r - g) < 5 and abs(g - b) < 5 and abs(r - b) < 5:  # Gris purs
                if r < 100:  # Gris trop sombres
                    continue
            
            all_colors.add(tuple(color))
    
    # Convertir en liste et trier
    letter_colors = sorted(list(all_colors))
    
    print(f"✅ {len(letter_colors)} couleurs uniques extraites")
    
    # Afficher quelques exemples
    print("🎨 Exemples de couleurs détectées:")
    for i, color in enumerate(letter_colors[:10]):  # Afficher les 10 premières
        print(f"  {i+1}: RGB{color}")
    
    if len(letter_colors) > 10:
        print(f"  ... et {len(letter_colors) - 10} autres couleurs")
    
    return letter_colors


def detect_text_with_tesseract(image_path):
    """
    Détecte le texte dans une image en utilisant Tesseract OCR.
    
    Args:
        image_path: Chemin vers l'image à analyser
    
    Returns:
        Texte détecté (string)
    """
    try:
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            return "Erreur: Impossible de charger l'image"
        
        # Convertir en niveaux de gris si nécessaire
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        # Améliorer l'image pour l'OCR
        # Appliquer un seuillage pour avoir du texte noir sur fond blanc
        _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        
        # Inverser si nécessaire (Tesseract fonctionne mieux avec texte noir sur fond blanc)
        # Compter les pixels blancs vs noirs pour décider
        white_pixels = np.sum(img_thresh == 255)
        black_pixels = np.sum(img_thresh == 0)
        
        if black_pixels > white_pixels:
            # Plus de pixels noirs que blancs, probablement texte blanc sur fond noir
            img_thresh = cv2.bitwise_not(img_thresh)
        
        # Configuration Tesseract pour reconnaître du texte simple
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_:-()'
        
        # Extraire le texte
        text = pytesseract.image_to_string(img_thresh, config=custom_config).strip()
        
        return text if text else "Aucun texte détecté"
        
    except Exception as e:
        return f"Erreur Tesseract: {str(e)}"


def color_based_letter_detector(image_path, output_dir, sample_colors_dir=None):
    """
    Détecte et découpe les lettres basé sur leur couleur spécifique.
    
    Args:
        image_path: Chemin vers l'image à analyser
        output_dir: Dossier de sortie pour les lettres
        sample_colors_dir: Dossier avec images d'exemple pour extraire les couleurs (optionnel)
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return 0
    
    # Convertir BGR vers RGB pour correspondre aux couleurs données
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Obtenir les couleurs des lettres
    if sample_colors_dir and os.path.exists(sample_colors_dir):
        letter_colors = extract_letter_colors_from_samples(sample_colors_dir)
        if not letter_colors:
            print("⚠️  Aucune couleur extraite, utilisation des couleurs par défaut")
            letter_colors = [
                (213, 212, 220),
                (215, 212, 239),
                (225, 228, 233),
                (197, 201, 230),
                (238, 227, 243)
            ]
    else:
        # Couleurs par défaut si pas d'échantillons
        letter_colors = [
            (213, 212, 220),
            (215, 212, 239),
            (225, 228, 233),
            (197, 201, 230),
            (238, 227, 243)
        ]
    
    # Créer un masque pour détecter toutes les couleurs de lettres
    height, width = img_rgb.shape[:2]
    letter_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Tolérance pour la détection de couleur
    tolerance = 15
    
    for color in letter_colors:
        # Créer un masque pour cette couleur spécifique
        lower_bound = np.array([max(0, c - tolerance) for c in color])
        upper_bound = np.array([min(255, c + tolerance) for c in color])
        
        # Détecter les pixels de cette couleur
        color_mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        
        # Ajouter au masque global
        letter_mask = cv2.bitwise_or(letter_mask, color_mask)
    
    # Sauvegarder le masque de debug
    mask_debug_path = os.path.join(output_dir, "color_mask_debug.png")
    cv2.imwrite(mask_debug_path, letter_mask)
    
    # Détecter le texte avec Tesseract sur le masque
    detected_text = detect_text_with_tesseract(mask_debug_path)
    
    # Sauvegarder le texte détecté
    text_output_path = os.path.join(output_dir, "tesseract_detected_text.txt")
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Texte détecté par Tesseract: {detected_text}\n")
        f.write(f"Fichier source: {os.path.basename(image_path)}\n")
    
    print(f"📝 Texte détecté par Tesseract: '{detected_text}'")
    
    # Calculer la projection verticale du masque
    vertical_projection = np.sum(letter_mask, axis=0)
    
    # Trouver les zones de lettres
    threshold = height * 0.05  # Seuil plus bas car on détecte spécifiquement les lettres
    
    in_letter = False
    letter_start = 0
    letter_positions = []
    
    for x in range(width):
        if vertical_projection[x] > threshold:  # Il y a des pixels de lettre
            if not in_letter:
                letter_start = x
                in_letter = True
        else:  # Pas de pixels de lettre
            if in_letter:
                # Fin d'une lettre
                letter_positions.append((letter_start, x))
                in_letter = False
    
    # Gérer le cas où la dernière lettre va jusqu'au bord
    if in_letter:
        letter_positions.append((letter_start, width))
    
    # Filtrer les lettres trop petites
    min_letter_width = 3
    letter_positions = [(start, end) for start, end in letter_positions if end - start >= min_letter_width]
    
    letters_saved = 0
    
    # Créer une image de debug avec les détections
    debug_img = img.copy()
    
    # Découper et sauvegarder chaque lettre
    for i, (start_x, end_x) in enumerate(letter_positions):
        # Ajouter un petit padding horizontal
        padding = 2
        x_start = max(0, start_x - padding)
        x_end = min(width, end_x + padding)
        
        # Calculer les limites verticales pour cette lettre
        letter_mask_crop = letter_mask[:, x_start:x_end]
        horizontal_projection = np.sum(letter_mask_crop, axis=1)
        
        # Trouver les limites verticales (haut et bas de la lettre)
        y_positions = np.where(horizontal_projection > 0)[0]
        
        if len(y_positions) > 0:
            y_start = max(0, y_positions[0] - padding)
            y_end = min(height, y_positions[-1] + padding + 1)
        else:
            # Fallback si pas de pixels détectés
            y_start = 0
            y_end = height
        
        # Extraire la lettre avec les bonnes dimensions
        letter_crop = img[y_start:y_end, x_start:x_end]
        
        # Dessiner les zones détectées sur l'image de debug (rectangle précis)
        cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(debug_img, str(i+1), (x_start, y_start - 5 if y_start > 15 else y_end + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Sauvegarder la lettre
        letter_filename = f"letter_{i+1:02d}.png"
        letter_path = os.path.join(output_dir, letter_filename)
        cv2.imwrite(letter_path, letter_crop)
        letters_saved += 1
    
    # Sauvegarder l'image de debug avec les détections
    debug_filename = "color_detection_debug.png"
    debug_path = os.path.join(output_dir, debug_filename)
    cv2.imwrite(debug_path, debug_img)
    
    # Créer une visualisation de la projection verticale
    projection_img = np.zeros((200, width, 3), dtype=np.uint8)
    max_projection = np.max(vertical_projection) if np.max(vertical_projection) > 0 else 1
    
    for x in range(width):
        bar_height = int((vertical_projection[x] / max_projection) * 180)
        cv2.line(projection_img, (x, 199), (x, 199 - bar_height), (255, 255, 255), 1)
    
    # Dessiner le seuil
    threshold_line = int((threshold / max_projection) * 180)
    cv2.line(projection_img, (0, 199 - threshold_line), (width, 199 - threshold_line), (0, 0, 255), 2)
    
    projection_path = os.path.join(output_dir, "color_projection_debug.png")
    cv2.imwrite(projection_path, projection_img)
    
    # Créer une image montrant les couleurs détectées
    colored_mask = np.zeros_like(img)
    for i, color in enumerate(letter_colors):
        lower_bound = np.array([max(0, c - tolerance) for c in color])
        upper_bound = np.array([min(255, c + tolerance) for c in color])
        color_mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        
        # Colorier chaque couleur différemment pour le debug
        debug_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i]
        colored_mask[color_mask > 0] = debug_color
    
    colored_debug_path = os.path.join(output_dir, "colors_detected_debug.png")
    cv2.imwrite(colored_debug_path, colored_mask)
    
    return letters_saved
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return 0

    
    letters_saved = 0
    
    return letters_saved


def process_zone_preview_files():
    """
    Double boucle sur les dossiers ./tmp et les fichiers zone_preview_player*_name.png
    Découpe chaque fichier en lettres individuelles
    """
    print("🔍 Découpage des fichiers zone_preview par détection de couleur...")
    
    # Première boucle : trouver tous les dossiers tmp
    tmp_pattern = "./tmp/*"
    tmp_folders = glob.glob(tmp_pattern)
    
    # Filtrer pour garder seulement les dossiers
    tmp_folders = [folder for folder in tmp_folders if os.path.isdir(folder)]
    tmp_folders.sort()  # Trier pour un ordre cohérent
    
    if not tmp_folders:
        print("❌ Aucun dossier trouvé dans ./tmp/")
        return
    
    print(f"📁 Dossiers trouvés: {len(tmp_folders)}")
    
    # Créer le dossier de sortie principal
    output_base_dir = Path("./extracted_letters")
    output_base_dir.mkdir(exist_ok=True)
    
    total_files_processed = 0
    total_letters_extracted = 0
    
    # Première boucle : parcourir chaque dossier tmp
    for folder_path in tmp_folders:
        folder_name = os.path.basename(folder_path)
        print(f"\n🎯 Dossier: {folder_name}")
        
        # Créer le dossier de sortie pour ce dossier
        folder_output_dir = output_base_dir / folder_name
        folder_output_dir.mkdir(exist_ok=True)
        
        # Deuxième boucle : traiter les fichiers zone_preview dans ce dossier
        target_files = [
            "zone_preview_player1_name.png",
            "zone_preview_player2_name.png"
        ]
        
        for filename in target_files:
            file_path = os.path.join(folder_path, filename)
            
            if os.path.exists(file_path):
                print(f"  📄 {filename} -> ", end="")
                
                # Créer le dossier de sortie pour ce fichier
                file_output_dir = folder_output_dir / filename.replace('.png', '')
                file_output_dir.mkdir(exist_ok=True)
                
                # Découper le fichier en lettres par détection de couleur
                letters_count = color_based_letter_detector(file_path, str(file_output_dir))
                
                print(f"{letters_count} lettres extraites")
                total_files_processed += 1
                total_letters_extracted += letters_count
                
            else:
                print(f"  ❌ {filename} (non trouvé)")
    
    print("\n✅ Terminé!")
    print(f"📊 Fichiers traités: {total_files_processed}")
    print(f"📊 Lettres extraites: {total_letters_extracted}")
    print(f"📁 Lettres sauvegardées dans: {output_base_dir}")
    print("\n🔍 Images de debug générées:")
    print("  - color_mask_debug.png : masque de détection global")
    print("  - color_detection_debug.png : rectangles verts sur lettres détectées")
    print("  - color_projection_debug.png : projection verticale")
    print("  - colors_detected_debug.png : chaque couleur avec une couleur différente")


def scan_tmp_folders():
    """
    Double boucle sur les dossiers ./tmp et les fichiers zone_preview_player*_name.png
    Affiche juste le nom des fichiers trouvés
    """
    print("🔍 Scan des dossiers tmp...")
    
    # Première boucle : trouver tous les dossiers tmp
    tmp_pattern = "./tmp/*"
    tmp_folders = glob.glob(tmp_pattern)
    
    # Filtrer pour garder seulement les dossiers
    tmp_folders = [folder for folder in tmp_folders if os.path.isdir(folder)]
    tmp_folders.sort()  # Trier pour un ordre cohérent
    
    if not tmp_folders:
        print("❌ Aucun dossier trouvé dans ./tmp/")
        return
    
    print(f"📁 Dossiers trouvés: {len(tmp_folders)}")
    
    # Première boucle : parcourir chaque dossier tmp
    for folder_path in tmp_folders:
        folder_name = os.path.basename(folder_path)
        print(f"\n🎯 Dossier: {folder_name}")
        
        # Deuxième boucle : chercher les fichiers zone_preview dans ce dossier
        target_files = [
            "zone_preview_player1_name.png",
            "zone_preview_player2_name.png"
        ]
        
        for filename in target_files:
            file_path = os.path.join(folder_path, filename)
            
            if os.path.exists(file_path):
                print(f"  📄 {filename}")
            else:
                print(f"  ❌ {filename} (non trouvé)")


def main():
    """Point d'entrée principal"""
    process_zone_preview_files()


if __name__ == "__main__":
    main()
