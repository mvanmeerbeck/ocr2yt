import cv2
from pathlib import Path


def test_text_detection_method_1(image, method_name="Seuillage adaptatif"):
    """
    Méthode 1: Seuillage adaptatif
    """
    print(f"\n--- {method_name} ---")

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Seuillage adaptatif
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Sauvegarder pour debug
    cv2.imwrite("tmp/method1_binary.png", binary)

    # Détection de contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return analyze_contours(contours, image, "method1")


def test_text_detection_method_2(image, method_name="Seuillage Otsu"):
    """
    Méthode 2: Seuillage Otsu
    """
    print(f"\n--- {method_name} ---")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Améliorer le contraste
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)

    # Seuillage Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite("tmp/method2_binary.png", binary)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return analyze_contours(contours, image, "method2")


def test_text_detection_method_3(image, method_name="Gradient morphologique"):
    """
    Méthode 3: Gradient morphologique
    """
    print(f"\n--- {method_name} ---")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gradient morphologique pour détecter les contours de texte
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # Seuillage
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fermeture pour connecter les caractères
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    cv2.imwrite("tmp/method3_binary.png", closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return analyze_contours(contours, image, "method3")


def test_text_detection_method_4(image, method_name="EAST Text Detection"):
    """
    Méthode 4: Détection de texte avec filtrage avancé
    """
    print(f"\n--- {method_name} ---")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Seuillage adaptatif avec paramètres différents
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10
    )

    # Opérations morphologiques pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("tmp/method4_binary.png", cleaned)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return analyze_contours(contours, image, "method4")


import cv2
import numpy as np
from pathlib import Path

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
    print("Tesseract disponible ✓")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract non disponible - installez avec: pip install pytesseract")


def extract_text_with_ocr(image_region, method_name="", region_info=""):
    """
    Extrait le texte d'une région d'image avec Tesseract
    """
    if not TESSERACT_AVAILABLE:
        return "OCR_NOT_AVAILABLE"

    if image_region.size == 0:
        return "EMPTY_REGION"

    try:
        # Configuration Tesseract pour noms de joueurs
        configs = {
            "default": "--psm 8",
            "single_word": "--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-",
            "strict": "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "lenient": "--psm 6",
        }

        results = {}

        # Preprocessing pour améliorer l'OCR
        gray = (
            cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            if len(image_region.shape) == 3
            else image_region
        )

        # Plusieurs techniques de preprocessing
        preprocessed_images = {
            "original": gray,
            "contrast": cv2.convertScaleAbs(gray, alpha=2.0, beta=0),
            "binary": cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[
                1
            ],
            "dilated": cv2.dilate(gray, np.ones((2, 2), np.uint8), iterations=1),
        }

        # Tester chaque combinaison preprocessing + config
        for prep_name, prep_image in preprocessed_images.items():
            for config_name, config in configs.items():
                try:
                    text = pytesseract.image_to_string(
                        prep_image, config=config
                    ).strip()
                    confidence = pytesseract.image_to_data(
                        prep_image, config=config, output_type=pytesseract.Output.DICT
                    )
                    avg_conf = np.mean(
                        [int(c) for c in confidence["conf"] if int(c) > 0]
                    )

                    if text and len(text) > 0:
                        key = f"{prep_name}_{config_name}"
                        results[key] = {
                            "text": text,
                            "confidence": avg_conf,
                            "length": len(text),
                            "clean_text": "".join(
                                c for c in text if c.isalnum() or c in "_-"
                            ),
                        }
                except Exception as e:
                    continue

        return results

    except Exception as e:
        return f"OCR_ERROR: {str(e)}"


def analyze_contours(contours, original_image, method_prefix):
    """
    Analyse les contours détectés et applique différents filtres
    """
    height, width = original_image.shape[:2]

    print(f"Contours détectés: {len(contours)}")

    # Créer une image pour visualiser tous les contours
    all_contours_img = original_image.copy()
    cv2.drawContours(all_contours_img, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(f"tmp/{method_prefix}_all_contours.png", all_contours_img)

    # Filtrer les contours
    text_candidates = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Différents niveaux de filtrage
        filters = {
            "basic": w > 10 and h > 5,
            "medium": w > 15 and h > 8 and w / h > 1.2,
            "strict": w > 20
            and h > 10
            and w / h > 1.5
            and w < width * 0.8
            and h < height * 0.6,
            "very_strict": w > 25
            and h > 12
            and 1.5 < w / h < 10
            and w < width * 0.7
            and h < height * 0.5,
        }

        for filter_name, condition in filters.items():
            if condition:
                text_candidates.append(
                    {
                        "filter": filter_name,
                        "bbox": (x, y, w, h),
                        "area": w * h,
                        "ratio": w / h,
                    }
                )

    # Grouper par filtre
    by_filter = {}
    for candidate in text_candidates:
        filter_name = candidate["filter"]
        if filter_name not in by_filter:
            by_filter[filter_name] = []
        by_filter[filter_name].append(candidate)

    # Créer des images pour chaque niveau de filtre
    for filter_name, candidates in by_filter.items():
        filter_img = original_image.copy()

        print(f"  Filtre {filter_name}: {len(candidates)} candidats")

        # Trier par taille
        candidates.sort(key=lambda x: x["area"], reverse=True)

        for i, candidate in enumerate(candidates[:3]):  # Top 3 pour OCR
            x, y, w, h = candidate["bbox"]

            # Couleur selon le rang
            colors = [
                (0, 0, 255),
                (0, 165, 255),
                (0, 255, 255),
            ]
            color = colors[i] if i < len(colors) else (128, 128, 128)

            cv2.rectangle(filter_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                filter_img,
                f"{i+1}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            print(f"    {i+1}. {w}x{h} (ratio: {candidate['ratio']:.1f})")

            # NOUVEAU: Extraire la région et faire de l'OCR
            region = original_image[y : y + h, x : x + w]
            if region.size > 0:
                # Sauvegarder la région pour debug
                region_path = f"tmp/{method_prefix}_{filter_name}_region_{i+1}.png"
                cv2.imwrite(region_path, region)

                # OCR sur cette région
                ocr_results = extract_text_with_ocr(
                    region, method_prefix, f"{filter_name}_{i+1}"
                )

                if isinstance(ocr_results, dict) and ocr_results:
                    # Trouver le meilleur résultat OCR
                    best_ocr = max(
                        ocr_results.items(), key=lambda x: x[1]["confidence"]
                    )
                    config_name, result = best_ocr

                    print(
                        f"       OCR: '{result['clean_text']}' (conf: {result['confidence']:.1f}%, {config_name})"
                    )

                    # Ajouter le texte sur l'image
                    text_to_show = result["clean_text"][:15]  # Limiter la longueur
                    cv2.putText(
                        filter_img,
                        text_to_show,
                        (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )
                else:
                    print(f"       OCR: {ocr_results}")

        cv2.imwrite(f"tmp/{method_prefix}_{filter_name}_filtered.png", filter_img)

    return by_filter


def test_all_methods_on_image(image_path):
    """
    Teste toutes les méthodes sur une image
    """
    print(f"\n=== ANALYSE DE: {image_path.name} ===")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Erreur: Impossible de charger {image_path}")
        return

    height, width = image.shape[:2]
    print(f"Dimensions: {width}x{height}")

    # Créer le dossier tmp
    Path("tmp").mkdir(exist_ok=True)

    # Sauvegarder l'image originale
    cv2.imwrite("tmp/original.png", image)

    # Tester toutes les méthodes
    results = {}
    results["method1"] = test_text_detection_method_1(image)
    results["method2"] = test_text_detection_method_2(image)
    results["method3"] = test_text_detection_method_3(image)
    results["method4"] = test_text_detection_method_4(image)

    # Résumé
    print(f"\n=== RÉSUMÉ POUR {image_path.name} ===")
    for method, filters in results.items():
        print(f"{method}:")
        for filter_name, candidates in filters.items():
            if candidates:
                best = max(candidates, key=lambda x: x["area"])
                print(
                    f"  {filter_name}: {len(candidates)} candidats, meilleur: {best['bbox'][2]}x{best['bbox'][3]}"
                )


def main():
    print("=== TEXT DETECTION DEBUGGER ===")

    # Chercher les fichiers zone_preview pour les noms
    tmp_dir = Path("./tmp")
    if not tmp_dir.exists():
        print("Dossier tmp/ non trouvé. Lancez d'abord zone_viewer.py")
        return

    # Chercher spécifiquement les fichiers zone_preview des noms
    target_files = [
        tmp_dir / "zone_preview_player1_name.png",
        tmp_dir / "zone_preview_player2_name.png",
    ]

    existing_files = [f for f in target_files if f.exists()]

    if not existing_files:
        print("Fichiers zone_preview_player*_name.png non trouvés dans tmp/")
        print("Lancez d'abord zone_viewer.py pour générer ces fichiers")
        return

    print(f"Fichiers de zones de noms trouvés:")
    for i, file in enumerate(existing_files):
        print(f"  {i+1}. {file.name}")

    # Si un seul fichier, l'utiliser directement
    if len(existing_files) == 1:
        selected_file = existing_files[0]
        print(f"Analyse automatique de: {selected_file.name}")
    else:
        # Demander à l'utilisateur de choisir
        try:
            choice = (
                int(input(f"\nChoisissez un fichier (1-{len(existing_files)}): ")) - 1
            )
            if 0 <= choice < len(existing_files):
                selected_file = existing_files[choice]
            else:
                print("Choix invalide")
                return
        except ValueError:
            print("Choix invalide")
            return

    # Analyser le fichier sélectionné
    test_all_methods_on_image(selected_file)

    print(f"\n=== FICHIERS GÉNÉRÉS ===")
    print("Regardez dans tmp/ pour voir tous les résultats:")
    print("- *_binary.png : Images binaires")
    print("- *_all_contours.png : Tous les contours détectés")
    print("- *_*_filtered.png : Contours filtrés par niveau")


if __name__ == "__main__":
    main()
