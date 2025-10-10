import cv2
from pathlib import Path

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


def show_zones_on_image(image_path):
    """
    Affiche les zones définies sur une image
    """
    # Charger l'image
    screenshot = cv2.imread(str(image_path))
    if screenshot is None:
        print(f"Erreur: Impossible de charger {image_path}")
        return

    height, width = screenshot.shape[:2]
    image_name = Path(image_path).stem  # Nom sans extension
    
    # Créer le dossier spécifique pour cette image
    tmp_dir = Path("./tmp") / image_name
    tmp_dir.mkdir(parents=True, exist_ok=True)

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

    # Créer une copie pour dessiner
    result_image = screenshot.copy()

    # Dessiner chaque zone
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

        # Déterminer la couleur selon le type de zone
        zone_type = zone_name.split("_")[-1]  # Dernier mot après underscore
        color = zone_colors.get(zone_type, (128, 128, 128))

        # Dessiner le rectangle de la zone
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)

        # Ajouter le label de la zone
        label_y = y - 10 if y > 30 else y + h + 25
        cv2.putText(
            result_image,
            zone_name,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Extraire et sauvegarder la zone pour inspection
        zone_image = screenshot[y : y + h, x : x + w]

        zone_path = tmp_dir / f"zone_preview_{zone_name}.png"
        cv2.imwrite(str(zone_path), zone_image)

    # Sauvegarder l'image avec les zones dans le dossier spécifique
    result_path = tmp_dir / "zones_preview.png"
    cv2.imwrite(str(result_path), result_image)

    # Afficher l'image
    cv2.namedWindow("Zones définies", cv2.WINDOW_NORMAL)
    cv2.imshow("Zones définies", result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():

    # Vérifier le dossier data
    data_folder = Path("./data/thumbnails/todo")
    if not data_folder.exists():
        print(f"Erreur: Le dossier '{data_folder}' n'existe pas.")
        print("Créez le dossier 'data' et ajoutez-y vos screenshots.")
        return

    # Lister les fichiers disponibles
    image_files = [
        f
        for f in data_folder.iterdir()
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
    ]

    if not image_files:
        print("Aucun fichier image trouvé dans './data'")
        return
    
    for image in image_files:
        show_zones_on_image(str(image))


if __name__ == "__main__":
    main()
