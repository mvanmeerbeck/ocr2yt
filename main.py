import cv2
import numpy as np
from pathlib import Path

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️  Tesseract non disponible - installez avec: pip install pytesseract")

# Configuration des zones de détection (coordonnées relatives à la taille de l'image)
GAME_ZONES = {
    # Zones pour le joueur 1 (gauche) - coordonnées en pourcentage (x, y, w, h)
    "player1_character": (0.05, 0.755, 0.23, 0.06),  # Zone character joueur 1
    "player1_rank": (0.02, 0.82, 0.085, 0.09),  # Zone rank joueur 1
    "player1_flag": (0.12, 0.84, 0.028, 0.034),  # Zone drapeau joueur 1
    "player1_name": (0.17, 0.84, 0.18, 0.038),  # Zone nom joueur 1
    # Zones pour le joueur 2 (droite) - coordonnées en pourcentage
    "player2_character": (0.72, 0.755, 0.23, 0.06),  # Zone character joueur 2
    "player2_rank": (0.892, 0.82, 0.085, 0.09),  # Zone rank joueur 2
    "player2_flag": (0.652, 0.84, 0.028, 0.034),  # Zone drapeau joueur 2
    "player2_name": (0.702, 0.84, 0.18, 0.038),  # Zone nom joueur 2
}


def load_templates_by_category(base_template_folder):
    """
    Charge les templates organisés par catégories dans des sous-dossiers
    """
    categories = {}

    if not base_template_folder.exists():
        return categories

    for category_folder in base_template_folder.iterdir():
        if category_folder.is_dir():
            category_name = category_folder.name
            templates_in_category = []

            print(f"Chargement de la catégorie: {category_name}")

            for template_file in category_folder.iterdir():
                if template_file.is_file():
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

    print(
        f"      Recherche zone {zone_image.shape} dans composite {composite_template.shape}"
    )

    # VOTRE LOGIQUE: Chercher la zone extraite DANS le composite
    result = cv2.matchTemplate(composite_template, zone_image, cv2.TM_CCOEFF_NORMED)

    # Trouver les positions où la correspondance dépasse le seuil
    locations = np.where(result >= threshold)

    zone_h, zone_w = zone_image.shape[:2]

    for pt in zip(*locations[::-1]):  # Switch x and y
        match_x, match_y = pt
        confidence = result[match_y, match_x]

        print(
            f"        Match trouvé à ({match_x}, {match_y}) avec confiance {confidence:.3f}"
        )

        # Déterminer quel template correspond à cette position
        for template_name, temp_x, temp_y, temp_w, temp_h in template_positions:
            # Vérifier si le match est dans ce template
            if (
                temp_x <= match_x <= temp_x + temp_w - zone_w
                and temp_y <= match_y <= temp_y + temp_h - zone_h
            ):

                matches.append((template_name, confidence, (match_x, match_y)))
                print(f"        -> Identifié comme template: {template_name}")
                break

    # Trier par confiance et retourner le meilleur
    matches = sorted(matches, key=lambda x: x[1], reverse=True)

    return matches


def analyze_screenshot_with_zones(screenshot, category_composites, threshold=0.6):
    """
    Analyse un screenshot en utilisant l'approche par zones
    """
    print("  Extraction des zones du screenshot...")
    zones = extract_zones_from_screenshot(screenshot)

    results = {
        "player1": {"character": None, "rank": None, "flag": None, "name": None},
        "player2": {"character": None, "rank": None, "flag": None, "name": None},
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
    }

    # Analyser chaque zone
    for zone_name, zone_data in zones.items():
        zone_image = zone_data["image"]

        # TRAITEMENT NORMAL POUR LES AUTRES ZONES : Template matching
        if zone_name not in zone_to_category:
            continue

        category = zone_to_category[zone_name]

        if category not in category_composites:
            print(f"    Catégorie '{category}' non trouvée pour zone '{zone_name}'")
            continue

        composite, positions = category_composites[category]

        print(f"    Analyse zone '{zone_name}' (catégorie: {category})")

        # Sauvegarder la zone pour debug
        tmp_dir = Path("./tmp")
        tmp_dir.mkdir(exist_ok=True)
        zone_debug_path = tmp_dir / f"debug_zone_{zone_name}.png"
        cv2.imwrite(str(zone_debug_path), zone_image)

        # Chercher cette zone dans le composite
        matches = find_zone_in_composite(zone_image, composite, positions, threshold)

        if matches:
            best_match = matches[0]  # Meilleur match
            template_name, confidence, match_coords = best_match

            print(f"      -> Trouvé: {template_name} (confiance: {confidence:.3f})")

            # Stocker le résultat selon la zone
            if "player1" in zone_name:
                if "character" in zone_name:
                    results["player1"]["character"] = template_name
                elif "rank" in zone_name:
                    results["player1"]["rank"] = template_name
                elif "flag" in zone_name:
                    results["player1"]["flag"] = template_name

            elif "player2" in zone_name:
                if "character" in zone_name:
                    results["player2"]["character"] = template_name
                elif "rank" in zone_name:
                    results["player2"]["rank"] = template_name
                elif "flag" in zone_name:
                    results["player2"]["flag"] = template_name
        else:
            print(f"      -> Aucun match trouvé")

    return results, zones


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


def main():
    # Chemins des dossiers
    data_folder = Path("./data/thumbnails")
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
        print(f"Dimensions: {width}x{height}")

        # NOUVELLE APPROCHE: Analyse par zones
        results, zones = analyze_screenshot_with_zones(
            screenshot, category_composites, threshold=0.6
        )

        # Afficher les résultats
        print("\n  === RÉSULTATS PAR JOUEUR ===")
        for player, info in results.items():
            if player.startswith("player"):
                print(f"  {player.upper()}:")
                for key, value in info.items():
                    print(f"    - {key}: {value or 'Non détecté'}")

        # Dessiner et afficher
        result_image = draw_zones_and_results(screenshot, zones, results)

        window_name = f"Screenshot {i+1}: {file_path.name}"
        cv2.imshow(window_name, result_image)

        print("Appuyez sur une touche pour continuer (ou 'q' pour quitter)...")
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyWindow(window_name)

        if key == ord("q"):
            print("Arrêt demandé par l'utilisateur.")
            break

    cv2.destroyAllWindows()
    print("Terminé!")


if __name__ == "__main__":
    main()
