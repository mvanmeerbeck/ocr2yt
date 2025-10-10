import cv2
import numpy as np
from pathlib import Path
import pytesseract
import os
import re
import json

# Configuration des zones de détection (coordonnées relatives à la taille de l'image)
GAME_ZONES = {
    # Zones pour le joueur 1 (gauche) - coordonnées en pourcentage (x, y, w, h)
    "player1_character": (0.05, 0.755, 0.23, 0.06),  # Zone character joueur 1
    "player1_rank": (0.02, 0.82, 0.085, 0.09),  # Zone rank joueur 1
    "player1_flag": (0.12, 0.84, 0.028, 0.034),  # Zone drapeau joueur 1
    "player1_name": (0.17, 0.84, 0.10, 0.038),  # Zone nom joueur 1
    "player1_control": (0.28, 0.84, 0.08, 0.038),  # Zone control joueur 1
    "player1_mr": (0.0124, 0.915, 0.031, 0.032),  # Zone master rank joueur 1
    "player1_lp": (0.064, 0.915, 0.042, 0.032),  # Zone league points joueur 1
    # Zones pour le joueur 2 (droite) - coordonnées en pourcentage
    "player2_character": (0.72, 0.755, 0.23, 0.06),  # Zone character joueur 2
    "player2_rank": (0.892, 0.82, 0.085, 0.09),  # Zone rank joueur 2
    "player2_flag": (0.652, 0.84, 0.028, 0.034),  # Zone drapeau joueur 2
    "player2_name": (0.702, 0.84, 0.10, 0.038),  # Zone nom joueur 2
    "player2_control": (0.50, 0.84, 0.08, 0.038),  # Zone control joueur 2
    "player2_mr": (0.884, 0.915, 0.031, 0.032),  # Zone master rank joueur 2
    "player2_lp": (0.936, 0.915, 0.042, 0.032),  # Zone league points joueur 2
}


def load_templates_by_category(base_template_folder):
    """
    Charge les templates organisés par catégories dans des sous-dossiers
    Supporte maintenant une structure hiérarchique pour les characters:
    - characters/akuma/akuma-1.png, akuma-2.png
    - ranks/master.png
    - flags/fr.png
    """
    categories = {}

    if not base_template_folder.exists():
        return categories

    for category_folder in base_template_folder.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            templates_in_category = []

            print(f"Chargement de la catégorie: {category_name}")

            # Structure spéciale pour les characters (avec sous-dossiers par personnage)
            if category_name == "characters":
                for character_folder in category_folder.iterdir():
                    if character_folder.is_dir():
                        character_name = character_folder.name
                        print(f"  Chargement du personnage: {character_name}")
                        
                        for template_file in character_folder.iterdir():
                            if template_file.is_file() and template_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                template_img = cv2.imread(str(template_file))
                                if template_img is not None:
                                    # Utiliser le nom du personnage comme template_name
                                    # (tous les templates d'un même personnage auront le même nom)
                                    templates_in_category.append((character_name, template_img))
                                    print(f"    - {template_file.name} chargé comme '{character_name}'")
                    else:
                        # Fichiers directement dans le dossier characters (ancien format)
                        if character_folder.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            template_img = cv2.imread(str(character_folder))
                            if template_img is not None:
                                template_name = character_folder.stem
                                templates_in_category.append((template_name, template_img))
                                print(f"  - {template_name} chargé (format direct)")
            else:
                # Structure normale pour les autres catégories (ranks, flags, etc.)
                for template_file in category_folder.iterdir():
                    if template_file.is_file() and template_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        template_img = cv2.imread(str(template_file))
                        if template_img is not None:
                            template_name = template_file.stem
                            templates_in_category.append((template_name, template_img))
                            print(f"  - {template_name} chargé")

            if templates_in_category:
                categories[category_name] = templates_in_category
                print(
                    f"  -> {len(templates_in_category)} template(s) dans {category_name}"
                )

    return categories


def create_composite_template(templates, spacing=10):
    """
    Crée un template composite en arrangeant plusieurs templates en grille
    """
    if not templates:
        return None, []

    template_info = []
    max_height = 0
    total_width = 0

    for name, template in templates:
        h, w = template.shape[:2]
        template_info.append((name, template, w, h))
        max_height = max(max_height, h)
        total_width += w

    total_width += spacing * (len(templates) - 1)
    composite = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    template_positions = []
    current_x = 0

    for name, template, w, h in template_info:
        y_offset = (max_height - h) // 2
        composite[y_offset : y_offset + h, current_x : current_x + w] = template
        template_positions.append((name, current_x, y_offset, w, h))
        current_x += w + spacing

    return composite, template_positions


def create_category_composites(template_categories, spacing=20):
    """
    Crée un template composite pour chaque catégorie
    """
    category_composites = {}

    for category_name, templates in template_categories.items():
        print(
            f"Création du composite pour '{category_name}' ({len(templates)} templates)..."
        )

        composite, positions = create_composite_template(templates, spacing)

        if composite is not None:
            category_composites[category_name] = (composite, positions)
            print(
                f"  -> Composite {category_name}: {composite.shape[1]}x{composite.shape[0]} pixels"
            )

            # Sauvegarder le composite pour debug
            debug_path = Path(f"./tmp/composite_{category_name}_debug.png")
            cv2.imwrite(str(debug_path), composite)
            print(f"  -> Sauvegardé: {debug_path}")
        else:
            print(f"  -> Échec création composite pour {category_name}")

    return category_composites


def detect_text_with_tesseract(image_path, debug_mode=True):
    """
    Détecte le texte dans une image en utilisant Tesseract OCR avec debug détaillé.
    
    Args:
        image_path: Chemin vers l'image à analyser
        debug_mode: Active le mode debug avec sauvegarde des étapes
    
    Returns:
        Texte détecté (string)
    """
    try:
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            return "Erreur: Impossible de charger l'image"
        
        # Préparer le dossier de debug
        debug_dir = None
        if debug_mode:
            debug_dir = os.path.dirname(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Convertir en niveaux de gris si nécessaire
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img
        
        if debug_mode:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_1_grayscale.png"), img_gray)
        
        # Améliorer l'image pour l'OCR
        # Appliquer un seuillage pour avoir du texte noir sur fond blanc
        _, img_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        
        if debug_mode:
            cv2.imwrite(os.path.join(debug_dir, f"{base_name}_2_threshold.png"), img_thresh)
        
        # Inverser si nécessaire (Tesseract fonctionne mieux avec texte noir sur fond blanc)
        # Compter les pixels blancs vs noirs pour décider
        white_pixels = np.sum(img_thresh == 255)
        black_pixels = np.sum(img_thresh == 0)
        
        inverted = False
        if black_pixels > white_pixels:
            # Plus de pixels noirs que blancs, probablement texte blanc sur fond noir
            img_thresh = cv2.bitwise_not(img_thresh)
            inverted = True
        
        # Épaisissement du texte noir (utiliser erode, pas dilate !)
        img_thresh = cv2.erode(img_thresh, np.ones((3, 3), np.uint8), iterations=1)
        
        # Créer plusieurs versions de l'image pour maximiser les chances
        images_to_test = []
        
        # 1. Image originale (après épaisissement)
        images_to_test.append(("original", img_thresh))
        
        if debug_mode:
            for name, img_test in images_to_test:
                cv2.imwrite(os.path.join(debug_dir, f"{base_name}_test_{name}.png"), img_test)
        
        # Tester TOUTES les configurations sur TOUTES les images
        configurations = [
            ("psm7_oem1", r'--oem 1 --psm 7'), 
        ]
        
        results = []
        best_text = ""
        best_config = ""
        best_image_type = ""
        
        # Tester CHAQUE configuration sur CHAQUE image
        for img_name, test_img in images_to_test:
            for config_name, config in configurations:
                try:
                    full_test_name = f"{config_name}_{img_name}"
                    text = pytesseract.image_to_string(test_img, config=config).strip()
                    
                    # Nettoyer le texte
                    if text:
                        text = text.replace('\n', ' ').replace('\t', ' ')
                        text = re.sub(r'\s+', ' ', text).strip()
                    
                    results.append((full_test_name, config, text, len(text)))
                    
                    # Choisir le meilleur résultat (le plus long non vide)
                    if len(text) > len(best_text):
                        best_text = text
                        best_config = full_test_name
                        best_image_type = img_name
                        
                    if debug_mode:
                        print(f"    {full_test_name}: '{text}' (longueur: {len(text)})")
                        
                except Exception as e:
                    full_test_name = f"{config_name}_{img_name}"
                    results.append((full_test_name, config, f"Erreur: {str(e)}", 0))
                    if debug_mode:
                        print(f"    {full_test_name}: ERREUR - {str(e)}")
        
        # Sauvegarder les résultats de debug
        if debug_mode:
            debug_file = os.path.join(debug_dir, f"{base_name}_tesseract_debug.txt")
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"=== DEBUG TESSERACT ===\n")
                f.write(f"Image source: {image_path}\n")
                f.write(f"Dimensions: {img.shape}\n")
                f.write(f"Pixels blancs: {white_pixels}, noirs: {black_pixels}\n")
                f.write(f"Image inversée: {inverted}\n")
                f.write(f"Meilleure config: {best_config}\n")
                f.write(f"Meilleur résultat: '{best_text}'\n\n")
                
                f.write("=== TOUS LES RÉSULTATS ===\n")
                for config_name, config, text, length in results:
                    f.write(f"{config_name}: '{text}' (longueur: {length})\n")
                    f.write(f"  Config: {config}\n\n")
            
            print(f"    Debug sauvegardé: {debug_file}")
        
        return best_text if best_text else "Aucun texte détecté"
        
    except Exception as e:
        return f"Erreur Tesseract: {str(e)}"


def color_based_name_detector(zone_image, zone_name, debug_dir=None):
    """
    Détecte le nom d'un joueur en utilisant la détection de couleur spécifique aux lettres.
    Adapté de generate_templates.py pour être intégré dans main.py
    
    Args:
        zone_image: Image de la zone nom extraite
        zone_name: Nom de la zone (pour debug)
        debug_dir: Dossier pour sauvegarder les fichiers de debug
    
    Returns:
        Texte détecté (string)
    """
    if zone_image is None:
        return "Erreur: Image vide"
    
    # Convertir BGR vers RGB pour correspondre aux couleurs données
    img_rgb = cv2.cvtColor(zone_image, cv2.COLOR_BGR2RGB)
    
    # Couleurs spécifiques aux lettres des noms (adaptées du fichier generate_templates.py)
    letter_colors = [
        (213, 212, 220),
        (215, 212, 239),
        (225, 228, 233),
        (197, 201, 230),
        (238, 227, 243),
        (183, 187, 212),
        (183, 189, 213),
        (210, 182, 197),
        (199, 175, 189),
        (212, 188, 202)
    ]
    
    # Créer un masque pour détecter toutes les couleurs de lettres
    height, width = img_rgb.shape[:2]
    letter_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Tolérance pour la détection de couleur
    tolerance = 15
    
    for color in letter_colors:
        # Créer un masque pour cette couleur spécifique
        lower_bound = np.array([max(0, c - tolerance) for c in color], dtype=np.uint8)
        upper_bound = np.array([min(255, c + tolerance) for c in color], dtype=np.uint8)
        
        # Détecter les pixels de cette couleur
        color_mask = cv2.inRange(img_rgb, lower_bound, upper_bound)
        
        # Ajouter au masque global
        letter_mask = cv2.bitwise_or(letter_mask, color_mask)
    
    # Sauvegarder le masque de debug si un dossier est fourni
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        mask_debug_path = os.path.join(debug_dir, f"{zone_name}_color_mask_debug.png")
        cv2.imwrite(mask_debug_path, letter_mask)
        
        # Sauvegarder aussi l'image originale de la zone pour debug
        zone_debug_path = os.path.join(debug_dir, f"{zone_name}_original_zone.png")
        cv2.imwrite(zone_debug_path, zone_image)
        
        print(f"    🔍 Debug détaillé pour {zone_name}...")
        
        # Tester la détection sur le masque de couleur
        print("    Test 1: Masque de couleur")
        detected_text_mask = detect_text_with_tesseract(mask_debug_path, debug_mode=True)
        
        # Comparer les deux résultats
        print(f"    Masque couleur: '{detected_text_mask}'")

        detected_text = detected_text_mask
        best_method = "masque de couleur"
        
        # Sauvegarder le résumé de tous les tests
        summary_path = os.path.join(debug_dir, f"{zone_name}_detection_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=== RÉSUMÉ DÉTECTION DE TEXTE ===\n")
            f.write(f"Zone: {zone_name}\n")
            f.write(f"Meilleure méthode: {best_method}\n")
            f.write(f"Résultat final: '{detected_text}'\n\n")
            f.write(f"Résultat masque couleur: '{detected_text_mask}'\n")
            
            # Analyser le masque de couleur
            mask_pixels = np.sum(letter_mask > 0)
            total_pixels = letter_mask.shape[0] * letter_mask.shape[1]
            coverage = (mask_pixels / total_pixels) * 100
            
            f.write(f"=== ANALYSE MASQUE COULEUR ===\n")
            f.write(f"Pixels détectés: {mask_pixels} / {total_pixels}\n")
            f.write(f"Couverture: {coverage:.1f}%\n")
            f.write(f"Couleurs cherchées: {letter_colors}\n")
            f.write(f"Tolérance: {tolerance}\n")
            
            if coverage < 5:
                f.write("⚠️  PROBLÈME: Très peu de pixels détectés par le masque couleur!\n")
                f.write("   Les couleurs de détection sont peut-être incorrectes.\n")
            elif coverage > 50:
                f.write("⚠️  PROBLÈME: Trop de pixels détectés par le masque couleur!\n") 
                f.write("   La tolérance est peut-être trop élevée.\n")
        
        return detected_text
    else:
        # Si pas de debug, créer un fichier temporaire pour Tesseract
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, letter_mask)
            detected_text = detect_text_with_tesseract(tmp_file.name)
            os.unlink(tmp_file.name)  # Supprimer le fichier temporaire
            return detected_text


def extract_zones_from_screenshot(screenshot):
    """
    Extrait les zones définies du screenshot
    """
    height, width = screenshot.shape[:2]
    extracted_zones = {}

    for zone_name, (x_pct, y_pct, w_pct, h_pct) in GAME_ZONES.items():
        # Convertir les pourcentages en pixels
        x = int(x_pct * width)
        y = int(y_pct * height)
        w = int(w_pct * width)
        h = int(h_pct * height)

        # Vérifier les limites
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        # Extraire la zone
        zone_image = screenshot[y : y + h, x : x + w]
        extracted_zones[zone_name] = {
            "image": zone_image,
            "coords": (x, y, w, h),
            "relative_coords": (x_pct, y_pct, w_pct, h_pct),
        }

    return extracted_zones


def find_zone_in_composite(
    zone_image, composite_template, template_positions, threshold=0.7
):
    """
    Cherche une zone extraite dans un template composite (VOTRE APPROCHE)

    Args:
        zone_image: Image de la zone extraite du screenshot
        composite_template: Template composite où chercher
        template_positions: Positions des sous-templates
        threshold: Seuil de confiance

    Returns:
        list: [(template_name, confidence, match_coords_in_composite), ...]
    """
    if zone_image is None or composite_template is None:
        return []

    matches = []

    # VOTRE LOGIQUE: Chercher la zone extraite DANS le composite
    result = cv2.matchTemplate(composite_template, zone_image, cv2.TM_CCOEFF_NORMED)

    # Trouver les positions où la correspondance dépasse le seuil
    locations = np.where(result >= threshold)

    zone_h, zone_w = zone_image.shape[:2]

    for pt in zip(*locations[::-1]):  # Switch x and y
        match_x, match_y = pt
        confidence = result[match_y, match_x]

        # Déterminer quel template correspond à cette position
        for template_name, temp_x, temp_y, temp_w, temp_h in template_positions:
            # Vérifier si le match est dans ce template
            if (
                temp_x <= match_x <= temp_x + temp_w - zone_w
                and temp_y <= match_y <= temp_y + temp_h - zone_h
            ):

                matches.append((template_name, confidence, (match_x, match_y)))
                break

    # Trier par confiance et retourner le meilleur
    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    return matches


def analyze_screenshot_with_zones(screenshot, category_composites, threshold=0.6):
    """
    Analyse un screenshot en utilisant l'approche par zones
    Intègre maintenant l'OCR avec la classe SimpleTemplateOCR pour plusieurs catégories
    """
    zones = extract_zones_from_screenshot(screenshot)

    results = {
        "player1": {"character": None, "rank": None, "flag": None, "name": None, "control": None, "mr": None, "lp": None},
        "player2": {"character": None, "rank": None, "flag": None, "name": None, "control": None, "mr": None, "lp": None},
        "game_info": {"timer": None, "round": None},
    }

    # Mappage zone -> catégorie de template
    zone_to_category = {
        "player1_character": "characters",
        "player2_character": "characters",
        "player1_rank": "ranks",
        "player2_rank": "ranks",
        "player1_flag": "flags",
        "player2_flag": "flags",
        "player1_control": "controls",
        "player2_control": "controls",
        "player1_name": "names",
        "player2_name": "names",
        "player1_mr": "mr_text",
        "player2_mr": "mr_text",
        "player1_lp": "lp_text",
        "player2_lp": "lp_text",
    }

    # Créer le dossier de debug
    tmp_dir = Path("./tmp")
    tmp_dir.mkdir(exist_ok=True)

    # Analyser chaque zone
    for zone_name, zone_data in zones.items():
        zone_image = zone_data["image"]
        
        # Traitement spécial pour les zones de noms, mr et lp avec détection de couleur
        if any(field in zone_name for field in ["name", "mr", "lp"]):
            print(f"    {zone_name}: Utilisation de la détection de couleur...")
            detected_text = color_based_name_detector(zone_image, zone_name, debug_dir=str(tmp_dir))
            
            # Stocker le résultat selon la zone
            if "player1" in zone_name:
                if "name" in zone_name:
                    results["player1"]["name"] = detected_text
                elif "mr" in zone_name:
                    results["player1"]["mr"] = detected_text
                elif "lp" in zone_name:
                    results["player1"]["lp"] = detected_text
            elif "player2" in zone_name:
                if "name" in zone_name:
                    results["player2"]["name"] = detected_text
                elif "mr" in zone_name:
                    results["player2"]["mr"] = detected_text
                elif "lp" in zone_name:
                    results["player2"]["lp"] = detected_text
                
            print(f"    {zone_name}: '{detected_text}' (détection couleur + Tesseract)")
            continue
        
        # Traitement normal pour les autres zones (character, rank, flag, control)
        category = zone_to_category.get(zone_name)
        
        if not category or category not in category_composites:
            print(f"    Catégorie '{category}' non trouvée pour zone '{zone_name}'")
            continue

        composite, positions = category_composites[category]

        # Sauvegarder la zone pour debug
        zone_debug_path = tmp_dir / f"debug_zone_{zone_name}.png"
        cv2.imwrite(str(zone_debug_path), zone_image)

        # Chercher cette zone dans le composite
        matches = find_zone_in_composite(zone_image, composite, positions, threshold)

        if matches:
            best_match = matches[0]  # Meilleur match
            template_name, confidence, match_coords = best_match

            # Stocker le résultat selon la zone
            if "player1" in zone_name:
                if "character" in zone_name:
                    results["player1"]["character"] = template_name
                elif "rank" in zone_name:
                    results["player1"]["rank"] = template_name
                elif "flag" in zone_name:
                    results["player1"]["flag"] = template_name
                elif "control" in zone_name:
                    results["player1"]["control"] = template_name

            elif "player2" in zone_name:
                if "character" in zone_name:
                    results["player2"]["character"] = template_name
                elif "rank" in zone_name:
                    results["player2"]["rank"] = template_name
                elif "flag" in zone_name:
                    results["player2"]["flag"] = template_name
                elif "control" in zone_name:
                    results["player2"]["control"] = template_name

            print(f"    {zone_name}: {template_name} (confiance: {confidence:.3f})")
        else:
            print("      -> Aucun match trouvé")

    return results, zones


def extract_video_id_from_filename(filename):
    """
    Extrait l'ID de la vidéo à partir du nom de fichier.
    Supporte différents formats: video_id.ext, video_id_suffix.ext, etc.
    
    Args:
        filename: Nom du fichier (avec ou sans extension)
    
    Returns:
        ID de la vidéo (string)
    """
    # Supprimer l'extension
    name_without_ext = Path(filename).stem
    
    # Pour les fichiers YouTube, l'ID peut contenir des underscores
    # Exemples: 
    # - "Kz8a9hm_DZU.jpg" -> "Kz8a9hm_DZU"
    # - "simple_video.jpg" -> "simple_video"
    # - "abc123.jpg" -> "abc123"
    
    # On garde le nom complet sans extension comme ID vidéo
    video_id = name_without_ext
    
    return video_id


def save_metadata_to_json(results, zones, video_id, screenshot_info, output_dir="./metadata/todo"):
    """
    Sauvegarde les résultats de détection en format JSON.
    
    Args:
        results: Dictionnaire des résultats de détection
        zones: Informations sur les zones extraites
        video_id: ID de la vidéo
        screenshot_info: Informations sur le screenshot (chemin, dimensions)
        output_dir: Dossier de sortie pour les métadonnées
    """
    # Créer le dossier metadata s'il n'existe pas
    metadata_dir = Path(output_dir)
    metadata_dir.mkdir(exist_ok=True)
    
    # Préparer les métadonnées dans le format demandé
    metadata = {
        "id": video_id,
        "players": [
            {
                "name": results.get("player1", {}).get("name"),
                "character": results.get("player1", {}).get("character"),
                "rank": results.get("player1", {}).get("rank"),
                "flag": results.get("player1", {}).get("flag"),
                "control": results.get("player1", {}).get("control"),
                "mr": results.get("player1", {}).get("mr"),
                "lp": results.get("player1", {}).get("lp")
            },
            {
                "name": results.get("player2", {}).get("name"),
                "character": results.get("player2", {}).get("character"),
                "rank": results.get("player2", {}).get("rank"),
                "flag": results.get("player2", {}).get("flag"),
                "control": results.get("player2", {}).get("control"),
                "mr": results.get("player2", {}).get("mr"),
                "lp": results.get("player2", {}).get("lp")
            }
        ]
    }
    
    # Chemin de sortie
    output_file = metadata_dir / f"{video_id}.json"
    
    # Sauvegarder en JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"    💾 Métadonnées sauvegardées: {output_file}")
        return str(output_file)
        
    except Exception as e:
        print(f"    ❌ Erreur lors de la sauvegarde des métadonnées: {e}")
        return None


def draw_zones_and_results(screenshot, zones, results):
    """
    Dessine les zones extraites et les résultats sur le screenshot
    """
    result_image = screenshot.copy()

    # Couleurs pour différents types de zones
    zone_colors = {
        "character": (0, 255, 0),  # Vert
        "rank": (255, 0, 0),  # Bleu
        "flag": (0, 0, 255),  # Rouge
        "name": (255, 255, 0),  # Cyan
        "control": (255, 165, 0),  # Orange
        "mr": (128, 0, 128),  # Violet
        "lp": (255, 20, 147),  # Rose foncé
        "timer": (255, 0, 255),  # Magenta
        "round_info": (0, 255, 255),  # Jaune
    }

    # Dessiner les zones
    for zone_name, zone_data in zones.items():
        x, y, w, h = zone_data["coords"]

        # Déterminer la couleur selon le type de zone
        zone_type = zone_name.split("_")[-1]  # Dernier mot après underscore
        color = zone_colors.get(zone_type, (128, 128, 128))

        # Dessiner le rectangle de la zone
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)

        # Ajouter le label de la zone
        cv2.putText(
            result_image, zone_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    # Ajouter les résultats en texte
    y_offset = 30
    for player, info in results.items():
        if player.startswith("player"):
            text = f"{player.upper()}:"
            cv2.putText(
                result_image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 25

            for key, value in info.items():
                if value:
                    text = f"  {key}: {value}"
                    cv2.putText(
                        result_image,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                    )
                    y_offset += 20
            y_offset += 10

    return result_image


def confirm_and_correct_fields(results, template_categories):
    """
    Interface en ligne de commande pour confirmer ou corriger chaque champ détecté.
    
    Args:
        results: Dictionnaire des résultats de détection
        template_categories: Dictionnaire des templates disponibles par catégorie
    
    Returns:
        Dictionnaire des résultats corrigés
    """
    corrected_results = {
        "player1": {"character": None, "rank": None, "flag": None, "name": None, "control": None, "mr": None, "lp": None},
        "player2": {"character": None, "rank": None, "flag": None, "name": None, "control": None, "mr": None, "lp": None},
        "game_info": {"timer": None, "round": None},
    }
    
    print("\n" + "="*60)
    print("🎮 VÉRIFICATION ET CORRECTION DES CHAMPS DÉTECTÉS")
    print("="*60)
    print("Pour chaque champ, vous pouvez :")
    print("  - Appuyer sur ENTRÉE pour accepter la valeur détectée")
    print("  - Taper une nouvelle valeur pour la corriger")
    print("  - Taper 'skip' pour ignorer ce champ")
    print("                    - Taper 'list' pour voir les options disponibles (character, rank, flag, control)")
    print("-"*60)
    
    # Créer des listes des options disponibles pour l'aide
    available_options = {}
    if "characters" in template_categories:
        available_options["character"] = [name for name, _ in template_categories["characters"]]
    if "ranks" in template_categories:
        available_options["rank"] = [name for name, _ in template_categories["ranks"]]
    if "flags" in template_categories:
        available_options["flag"] = [name for name, _ in template_categories["flags"]]
    if "controls" in template_categories:
        available_options["control"] = [name for name, _ in template_categories["controls"]]
    
    # Traiter chaque joueur
    for player_num in [1, 2]:
        player_key = f"player{player_num}"
        print(f"\n🔹 JOUEUR {player_num}")
        print("-" * 20)
        
        # Traiter chaque champ pour ce joueur
        for field in ["name", "character", "rank", "flag", "control", "mr", "lp"]:
            detected_value = results.get(player_key, {}).get(field)
            
            # Affichage de la valeur détectée
            if detected_value:
                prompt = f"  {field.capitalize()}: '{detected_value}' ✅"
            else:
                prompt = f"  {field.capitalize()}: (non détecté) ❌"
            
            # Demander confirmation/correction
            user_input = input(f"{prompt} → ").strip()
            
            # Traitement de la réponse
            if user_input == "":
                # Accepter la valeur détectée
                corrected_results[player_key][field] = detected_value
                print(f"    ✓ Accepté: '{detected_value}'")
                
            elif user_input.lower() == "skip":
                # Ignorer ce champ
                corrected_results[player_key][field] = None
                print("    ⏭️  Ignoré")
                
            elif user_input.lower() == "list" and field in available_options:
                # Afficher les options disponibles
                print(f"    📋 Options disponibles pour {field}:")
                options = available_options[field]
                for i, option in enumerate(options[:20]):  # Limiter à 20 pour l'affichage
                    print(f"      - {option}")
                if len(options) > 20:
                    print(f"      ... et {len(options) - 20} autres")
                
                # Redemander la valeur
                user_input = input(f"  {field.capitalize()}: → ").strip()
                if user_input == "":
                    corrected_results[player_key][field] = detected_value
                elif user_input.lower() == "skip":
                    corrected_results[player_key][field] = None
                else:
                    corrected_results[player_key][field] = user_input
                    print(f"    ✏️  Corrigé: '{user_input}'")
                    
            else:
                # Valeur corrigée par l'utilisateur
                corrected_results[player_key][field] = user_input
                print(f"    ✏️  Corrigé: '{user_input}'")
    
    # Affichage du résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ FINAL")
    print("="*60)
    for player_num in [1, 2]:
        player_key = f"player{player_num}"
        print(f"\n🔹 JOUEUR {player_num}:")
        for field in ["name", "character", "rank", "flag", "control", "mr", "lp"]:
            value = corrected_results[player_key][field]
            status = "✅" if value else "❌"
            print(f"  {field.capitalize()}: {value or '(vide)'} {status}")
    
    # Demander confirmation finale
    print("\n" + "-"*60)
    confirm = input("Sauvegarder ces résultats ? (O/n) : ").strip().lower()
    
    if confirm == "" or confirm in ['o', 'oui', 'y', 'yes']:
        print("✅ Résultats confirmés et sauvegardés!")
        return corrected_results
    else:
        print("❌ Sauvegarde annulée.")
        return None


def main():
    # Chemins des dossiers
    data_folder = Path("./data/thumbnails/todo")
    template_base_folder = Path("./assets/templates")

    # Vérifications
    if not data_folder.exists():
        print(f"Le dossier '{data_folder}' n'existe pas.")
        print("Créez le dossier 'data' et ajoutez-y des screenshots.")
        return

    all_files = [f for f in data_folder.iterdir() if f.is_file()]
    if not all_files:
        print(f"Aucun fichier trouvé dans le dossier '{data_folder}'.")
        return

    # Charger les templates par catégories
    print("=== CHARGEMENT DES TEMPLATES PAR CATÉGORIES ===")
    template_categories = load_templates_by_category(template_base_folder)

    if not template_categories:
        print("Aucune catégorie de templates trouvée.")
        print("Structure attendue:")
        print("  assets/templates/characters/")
        print("  assets/templates/ranks/")
        print("  assets/templates/flags/")
        return

    print(f"Catégories trouvées: {list(template_categories.keys())}")

    # Créer les composites par catégorie
    print("\n=== CRÉATION DES COMPOSITES PAR CATÉGORIE ===")
    category_composites = create_category_composites(template_categories, spacing=15)

    print(f"\n{len(category_composites)} composite(s) créé(s)")

    # Analyser chaque screenshot
    print(f"\n=== TRAITEMENT DE {len(all_files)} FICHIER(S) ===")

    for i, file_path in enumerate(all_files):
        print(f"\n--- Fichier {i+1}/{len(all_files)}: {file_path.name} ---")

        # Charger le screenshot
        screenshot = cv2.imread(str(file_path))
        if screenshot is None:
            print(f"Impossible de charger: {file_path.name}")
            continue

        height, width = screenshot.shape[:2]
        
        # Extraire l'ID de la vidéo à partir du nom de fichier
        video_id = extract_video_id_from_filename(file_path.name)
        print(f"    ID vidéo extrait: {video_id}")
        
        # Préparer les informations sur le screenshot
        screenshot_info = {
            "filename": file_path.name,
            "width": width,
            "height": height,
            "file_size": file_path.stat().st_size if file_path.exists() else 0
        }

        # NOUVELLE APPROCHE: Analyse par zones avec OCR multiple intégré
        results, zones = analyze_screenshot_with_zones(
            screenshot, category_composites, threshold=0.6
        )

        # Afficher les résultats initiaux
        print("\n  === RÉSULTATS DÉTECTÉS AUTOMATIQUEMENT ===")
        for player, info in results.items():
            if player.startswith("player"):
                print(f"  {player.upper()}:")
                for key, value in info.items():
                    print(f"    - {key}: {value or 'Non détecté'}")

        # Dessiner et afficher l'image avec les zones
        result_image = draw_zones_and_results(screenshot, zones, results)

        window_name = f"Screenshot {i+1}: {file_path.name}"
        cv2.imshow(window_name, result_image)
        cv2.waitKey(1)  # Petit délai pour s'assurer que l'image s'affiche
        
        print(f"\n📺 Image affichée: {window_name}")
        print("Regardez l'image pour vérifier les détections...")
        
        # Garder l'image affichée pendant la vérification
        # (l'utilisateur peut la consulter pendant qu'il répond aux questions)
        
        # Interface de confirmation/correction
        corrected_results = confirm_and_correct_fields(results, template_categories)
        
        # Fermer la fenêtre d'image
        cv2.destroyWindow(window_name)
        
        if corrected_results is not None:
            # Sauvegarder les métadonnées corrigées en JSON
            save_metadata_to_json(corrected_results, zones, video_id, screenshot_info)
            
            # Déplacer l'image vers le dossier "done"
            done_folder = Path("./data/thumbnails/done")
            done_folder.mkdir(parents=True, exist_ok=True)
            
            destination_path = done_folder / file_path.name
            try:
                file_path.rename(destination_path)
                print(f"    📁 Image déplacée vers: {destination_path}")
            except Exception as e:
                print(f"    ⚠️  Erreur lors du déplacement: {e}")
        else:
            print("⏭️  Passage au fichier suivant sans sauvegarde.")

    cv2.destroyAllWindows()
    print("Terminé!")


if __name__ == "__main__":
    main()
