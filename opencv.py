import cv2
import numpy as np
import os
from typing import List, Tuple


class SimpleTemplateOCR:
    def __init__(self):
        """
        Système OCR simple basé sur templates fournis manuellement.
        """
        self.templates = {}  # Dict: caractère -> template (image numpy)
        self.template_info = {}  # Dict: caractère -> (width, height)

    def load_template(self, char: str, template_path: str):
        """
        Charge un template depuis un fichier image.

        Args:
            char: Le caractère que représente ce template ('N', 'i', 'c', 'K', 'y')
            template_path: Chemin vers l'image du template
        """
        # Vérifier que le caractère n'est pas vide
        if not char or not char.strip():
            print(f"❌ Caractère vide ignoré pour {template_path}")
            return False

        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            print(f"❌ Impossible de charger le template: {template_path}")
            return False

        self.templates[char] = template
        self.template_info[char] = {
            "width": template.shape[1],
            "height": template.shape[0],
        }

        return True

    def load_all_templates(self, templates_dir: str = "templates"):
        """
        Charge tous les templates depuis un dossier organisé par lettres.
        Chaque lettre a son dossier qui contient plusieurs templates.

        Args:
            templates_dir: Dossier contenant les sous-dossiers de templates
        """
        if not os.path.exists(templates_dir):
            print(f"❌ Dossier non trouvé: {templates_dir}")
            return False

        loaded_count = 0

        # Gérer les noms spéciaux pour la ponctuation
        char_mapping = {
            "colon": ":",
            "semicolon": ";",
            "slash": "/",
            "backslash": "\\",
            "question": "?",
            "star": "*",
            "less": "<",
            "greater": ">",
            "pipe": "|",
            "quote": '"',
            "apostrophe": "'",
            "space": " ",
            "underscore": "_",
            "dot": ".",
            "comma": ",",
        }

        # Parcourir tous les sous-dossiers
        for item in os.listdir(templates_dir):
            item_path = os.path.join(templates_dir, item)

            # Vérifier que c'est un dossier et qu'il n'est pas vide
            if os.path.isdir(item_path) and item.strip():
                # Le nom du dossier est le caractère (avec gestion des crochets)
                folder_name = item.strip()

                # Gérer les caractères entre crochets [N] -> N
                if folder_name.startswith("[") and folder_name.endswith("]"):
                    actual_char = folder_name[1:-1].strip()
                else:
                    # Utiliser le mapping si disponible, sinon le nom direct
                    actual_char = char_mapping.get(folder_name, folder_name)

                # Ignorer les caractères vides
                if not actual_char:
                    print(f"⚠️  Dossier ignoré: '{folder_name}' (caractère vide)")
                    continue

                # Charger tous les templates dans ce dossier
                template_files = []
                for filename in os.listdir(item_path):
                    if filename.endswith(".png"):
                        template_files.append(filename)

                # Trier les fichiers pour un ordre prévisible
                template_files.sort()

                # Charger chaque template de ce caractère
                char_loaded_count = 0
                for template_file in template_files:
                    template_path = os.path.join(item_path, template_file)

                    # Créer une clé unique pour chaque template: char + numéro
                    if "-" in template_file:
                        parts = template_file.split("-")
                        if len(parts) >= 2:
                            template_number = parts[1].split(".")[0]
                        else:
                            template_number = "1"
                    else:
                        template_number = "1"

                    # Vérifier que le numéro n'est pas vide
                    if not template_number:
                        template_number = "1"

                    template_key = f"{actual_char}_{template_number}"

                    if self.load_template(template_key, template_path):
                        char_loaded_count += 1
                        loaded_count += 1

                        # Aussi enregistrer sous la clé simple pour compatibilité
                        if (
                            char_loaded_count == 1
                        ):  # Premier template = template principal
                            self.templates[actual_char] = self.templates[template_key]
                            self.template_info[actual_char] = self.template_info[
                                template_key
                            ]

        return loaded_count > 0

    def find_matches(
        self, image_path: str, threshold: float = 0.8
    ) -> List[Tuple[str, int, int, float]]:
        """
        Trouve toutes les correspondances des templates dans l'image.
        Utilise tous les templates disponibles pour chaque caractère.

        Args:
            image_path: Chemin vers l'image à analyser
            threshold: Seuil de confiance (0-1)

        Returns:
            Liste de (caractère, x, y, score) triée par position x
        """
        # Charger l'image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Impossible de charger l'image: {image_path}")
            return []

        all_matches = []

        # Grouper les templates par caractère
        char_templates = {}
        for template_key, template in self.templates.items():
            # Méthode simple : chercher le dernier underscore
            last_underscore = template_key.rfind("_")

            if last_underscore > 0:  # Il y a un underscore ET il n'est pas au début
                # Template avec numéro: c_1, N_1, underscore devient __1
                char = template_key[:last_underscore]
            else:
                # Template simple: c, N, _, etc.
                char = template_key

            # Debug: afficher le parsing
            # print(f"🔍 Template key: '{template_key}' -> caractère: '{char}'")

            # Vérifier que le caractère final n'est pas vide
            if not char:
                print(f"⚠️  Caractère vide ignoré pour template_key: '{template_key}'")
                continue

            if char not in char_templates:
                char_templates[char] = []
            char_templates[char].append((template_key, template))

        # Tester chaque caractère avec tous ses templates
        for char, templates_list in char_templates.items():
            char_matches = []

            for template_key, template in templates_list:
                # Template matching
                result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

                # Trouver les pics au-dessus du seuil
                locations = np.where(result >= threshold)

                for pt in zip(*locations[::-1]):  # Switch x,y
                    x, y = pt
                    score = result[y, x]
                    char_matches.append((char, x, y, score, template_key))

            # Garder seulement les meilleures correspondances pour chaque position
            if char_matches:
                # Grouper par position approximative (tolérance de quelques pixels)
                position_groups = {}
                for match in char_matches:
                    char_val, x, y, score, template_key = match
                    # Créer une clé de position avec tolérance
                    pos_key = (x // 5 * 5, y // 5 * 5)  # Grouper par blocs de 5 pixels

                    if pos_key not in position_groups:
                        position_groups[pos_key] = []
                    position_groups[pos_key].append(match)

                # Pour chaque groupe de position, garder le meilleur score
                for group_matches in position_groups.values():
                    best_match = max(group_matches, key=lambda m: m[3])  # Max par score
                    char_val, x, y, score, template_key = best_match
                    all_matches.append((char_val, x, y, score))

            template_count = len(templates_list)
            final_count = len([m for m in all_matches if m[0] == char])
            # print(f"  '{char}': {final_count} matches (testé avec {template_count} templates)")

        # Trier par position X (gauche à droite)
        all_matches.sort(key=lambda m: m[1])

        return all_matches

    def filter_overlaps(
        self, matches: List[Tuple[str, int, int, float]], overlap_threshold: float = 0.3
    ) -> List[Tuple[str, int, int, float]]:
        """
        Filtre les correspondances qui se chevauchent en gardant la meilleure.

        Args:
            matches: Liste de (caractère, x, y, score)
            overlap_threshold: Seuil de chevauchement (0-1)

        Returns:
            Liste filtrée
        """
        if not matches:
            return []

        # Créer les bounding boxes
        boxes = []
        for char, x, y, score in matches:
            # Ignorer les caractères vides ou invalides
            if not char or char not in self.template_info:
                print(f"⚠️  Caractère ignoré: '{char}' (pas d'info de template)")
                continue

            template_info = self.template_info[char]
            w, h = template_info["width"], template_info["height"]
            boxes.append((x, y, x + w, y + h, score, char))

        # Trier par score décroissant (meilleur d'abord)
        boxes.sort(key=lambda b: b[4], reverse=True)

        selected = []

        for box in boxes:
            x1, y1, x2, y2, score, char = box

            # Vérifier le chevauchement avec les boîtes déjà sélectionnées
            overlaps = False
            for sel_box in selected:
                sx1, sy1, sx2, sy2, _, _ = sel_box

                # Calculer l'intersection
                ix1 = max(x1, sx1)
                iy1 = max(y1, sy1)
                ix2 = min(x2, sx2)
                iy2 = min(y2, sy2)

                if ix1 < ix2 and iy1 < iy2:
                    # Il y a intersection
                    intersection_area = (ix2 - ix1) * (iy2 - iy1)
                    box_area = (x2 - x1) * (y2 - y1)
                    overlap_ratio = intersection_area / box_area

                    if overlap_ratio > overlap_threshold:
                        overlaps = True
                        break

            if not overlaps:
                selected.append(box)

        # Reconvertir et trier par position X
        filtered = [(char, x1, y1, score) for x1, y1, x2, y2, score, char in selected]
        filtered.sort(key=lambda m: m[1])

        return filtered

    def recognize_text(self, image_path: str, threshold: float = 0.8) -> str:
        """
        Reconnaissance complète du texte.

        Args:
            image_path: Chemin vers l'image
            threshold: Seuil de confiance

        Returns:
            Texte reconnu
        """
        matches = self.find_matches(image_path, threshold)
        filtered_matches = self.filter_overlaps(matches)

        # Construire le texte final
        text = "".join([char for char, x, y, score in filtered_matches])

        return text

    def debug_visualization(
        self, image_path: str, output_path: str, threshold: float = 0.8
    ):
        """
        Crée une image de debug avec les détections.

        Args:
            image_path: Image source
            output_path: Image de sortie avec annotations
            threshold: Seuil de confiance
        """
        # Charger l'image en couleur
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Impossible de charger l'image: {image_path}")
            return

        matches = self.find_matches(image_path, threshold)
        filtered_matches = self.filter_overlaps(matches)

        # Dessiner les détections
        for char, x, y, score in filtered_matches:
            template_info = self.template_info[char]
            w, h = template_info["width"], template_info["height"]

            # Rectangle vert pour les bonnes détections
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Label avec caractère et score
            label = f"{char} ({score:.2f})"
            cv2.putText(
                img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

        # Sauvegarder
        cv2.imwrite(output_path, img)

    def save_detected_letters(
        self, image_path: str, threshold: float = 0.8
    ):
        """
        Sauvegarde chaque lettre détectée dans des fichiers séparés.
        
        Args:
            image_path: Chemin vers l'image source
            threshold: Seuil de confiance
        """
        from pathlib import Path
        
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Impossible de charger l'image: {image_path}")
            return

        # Obtenir le nom de l'image sans extension
        image_name = Path(image_path).stem
        
        # Créer le dossier de debug pour cette image
        debug_dir = Path("./debug") / image_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Trouver et filtrer les correspondances
        matches = self.find_matches(image_path, threshold)
        filtered_matches = self.filter_overlaps(matches)
        
        # Découper et sauvegarder chaque lettre
        for i, (char, x, y, score) in enumerate(filtered_matches, 1):
            if char in self.template_info:
                template_info = self.template_info[char]
                w, h = template_info["width"], template_info["height"]
                
                # Vérifier les limites de l'image
                x_end = min(x + w, img.shape[1])
                y_end = min(y + h, img.shape[0])
                x_start = max(0, x)
                y_start = max(0, y)
                
                # Découper la lettre
                letter_crop = img[y_start:y_end, x_start:x_end]
                
                # Nom du fichier: zone1.png, zone2.png, etc.
                letter_filename = debug_dir / f"zone{i}.png"
                
                # Sauvegarder la lettre
                cv2.imwrite(str(letter_filename), letter_crop)
            else:
                print(f"  ⚠️  Zone {i}: caractère '{char}' ignoré (pas d'info template)")

    def debug_visualization_with_letters(
        self, image_path: str, output_path: str, threshold: float = 0.8
    ):
        """
        Crée une image de debug avec les détections ET sauvegarde chaque lettre séparément.

        Args:
            image_path: Image source
            output_path: Image de sortie avec annotations
            threshold: Seuil de confiance
        """
        # Faire la visualisation debug classique
        self.debug_visualization(image_path, output_path, threshold)
        
        # Sauvegarder chaque lettre séparément
        self.save_detected_letters(image_path, threshold)


def test_generic():
    """Test générique pour tous les templates disponibles."""

    # Initialiser le système
    ocr = SimpleTemplateOCR()

    # Charger tous les templates disponibles
    if not ocr.load_all_templates("assets/templates/names"):
        print("❌ Impossible de charger les templates")
        print("💡 Veuillez fournir des fichiers .png dans le dossier templates/")
        print("   Exemple: templates/N.png, templates/i.png, etc.")
        return

    # Créer le dossier debug
    os.makedirs("debug", exist_ok=True)

    # Images de test (on utilise les images originales ET les préprocessées)
    test_cases = [
        # Images originales
        ("data/texts/zone_preview_player1_name.png", None),  # None = reconnaissance libre
        ("data/texts/zone_preview_player2_name.png", None),
    ]

    for i, (test_image, expected) in enumerate(test_cases, 1):
        if os.path.exists(test_image):

            best_result = ""
            best_threshold = 0
            best_score_info = ""

            # Tester avec différents seuils
            for threshold in [0.85]:
                result = ocr.recognize_text(test_image, threshold)

                if expected is None:
                    # Mode reconnaissance libre - on affiche tous les résultats
                    # print(f"  Seuil {threshold}: '{result}'")
                    if (
                        result and not best_result
                    ):  # Prendre le premier résultat non-vide
                        best_result = result
                        best_threshold = threshold
                        best_score_info = f" (premier résultat non-vide)"
                elif result == expected:
                    print(f"🎉 PARFAIT avec seuil {threshold}!")
                    best_result = result
                    best_threshold = threshold
                    best_score_info = " (match parfait)"
                    break
                else:
                    pass  # print(f"  Seuil {threshold}: '{result}' ≠ '{expected}'")

            if expected is None:
                if best_result:
                    print(f"{os.path.basename(test_image)}: {best_result}")
                else:
                    print("⚠️  Aucun texte détecté")
            elif not best_result or best_result != expected:
                print("⚠️  Aucun seuil ne donne le résultat attendu")

            # Créer une image de debug avec le meilleur seuil (ou 0.8 par défaut)
            debug_threshold = best_threshold if best_threshold > 0 else 0.8
            debug_path = (
                f"debug/{os.path.basename(test_image).replace('.png', '_debug.png')}"
            )
            # Utiliser la nouvelle méthode qui sauvegarde aussi les lettres individuellement
            ocr.debug_visualization_with_letters(test_image, debug_path, debug_threshold)

        else:
            print(f"❌ Image de test non trouvée: {test_image}")


if __name__ == "__main__":
    test_generic()
