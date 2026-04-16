import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from pathlib import Path
import os
import random

original_data = pd.read_csv('./etiquetas.csv')
os.makedirs('./dataset_shrouded', exist_ok=True)

def apply_random_blur_patches(img_pil, num_patches=3, blur_strength=15):
    """
    Aplica parches de blur aleatorio (gris transparente) a diferentes partes de la imagen.

    Args:
        img_pil: imagen PIL
        num_patches: número de parches blur a aplicar (aleatorio entre 1 y este valor)
        blur_strength: intensidad del blur (radio en píxeles)
    """
    img_array = np.array(img_pil).astype(float)
    w, h = img_pil.size

    # Número aleatorio de parches (entre 1 y num_patches)
    actual_patches = random.randint(1, num_patches)

    for _ in range(actual_patches):
        # Tamaño aleatorio del parche (entre 15% y 40% del ancho)
        patch_size = random.randint(int(w * 0.15), int(w * 0.4))

        # Posición aleatoria
        x_start = random.randint(0, max(0, w - patch_size))
        y_start = random.randint(0, max(0, h - patch_size))
        x_end = min(w, x_start + patch_size)
        y_end = min(h, y_start + patch_size)

        # Extraer región
        patch_region = img_array[y_start:y_end, x_start:x_end, :].copy()

        # Convertir a PIL, aplicar blur y convertir de vuelta
        patch_pil = Image.fromarray(patch_region.astype(np.uint8))
        patch_blurred = patch_pil.filter(ImageFilter.GaussianBlur(radius=blur_strength))
        patch_blurred_array = np.array(patch_blurred).astype(float)

        # Crear overlay gris transparente
        gray_overlay = np.full_like(patch_blurred_array, 128, dtype=float)

        # Blendear: 40% gris transparente + 60% blur
        blended = (gray_overlay * 0.4 + patch_blurred_array * 0.6).astype(np.uint8)

        # Aplicar al array original
        img_array[y_start:y_end, x_start:x_end, :] = blended

    return Image.fromarray(img_array.astype(np.uint8))

def apply_zoomout_fade(img_pil):
    """
    Aplica transformación shrouded a la imagen:
    - Zoom out 70% (reduce tamaño a 70% del original)
    - Reduce color a 30% (más monocromático)
    - Aumenta transparencia (blending 50% con fondo gris)
    - Añade parches de blur gris aleatorio
    - Centra en canvas gris neutro
    """
    w, h = img_pil.size
    
    # ZOOM OUT: reducir a 70% del tamaño original
    zoom_w, zoom_h = int(w * 0.7), int(h * 0.7)
    img_resized = img_pil.resize((zoom_w, zoom_h), Image.Resampling.LANCZOS)

    # REDUCIR COLOR: aplicar desaturación (30% del color original)
    enhancer_color = ImageEnhance.Color(img_resized)
    img_faded = enhancer_color.enhance(0.3)

    # AUMENTAR TRANSPARENCIA: blendear con fondo gris (50% opacidad)
    bg_color = (128, 128, 128)
    img_transparent = Image.blend(Image.new("RGB", img_resized.size, bg_color), img_faded, alpha=0.5)

    # AÑADIR BLUR ALEATORIO: parches gris transparente en posiciones aleatorias
    img_with_blur = apply_random_blur_patches(img_transparent, num_patches=4, blur_strength=12)

    # CENTRAR en canvas con fondo neutro
    canvas = Image.new("RGB", (w, h), bg_color)
    canvas.paste(img_with_blur, ((w - zoom_w) // 2, (h - zoom_h) // 2))

    return canvas

def load_and_process_images():
    """Carga todas las imágenes, aplica transformación shrouded y las guarda"""
    dataset_dir = Path("./dataset")
    shrouded_dir = Path("./dataset_shrouded")

    processed_count = 0
    error_count = 0

    print(f"=" * 70)
    print(f"GENERANDO IMÁGENES SHROUDED")
    print(f"=" * 70)
    print(f"Transformación:")
    print(f"  ✓ Zoom out 70%")
    print(f"  ✓ Color 30% (desaturación)")
    print(f"  ✓ Transparencia 50% (blend gris)")
    print(f"  ✓ Blur aleatorio (1-4 parches, tamaño 15-40%, gris transparente)")
    print(f"Entrada: {dataset_dir.absolute()}")
    print(f"Salida: {shrouded_dir.absolute()}")
    print(f"Procesando {len(original_data)} imágenes...\n")

    for idx, row in original_data.iterrows():
        image_name = row['archivo']
        image_path = dataset_dir / image_name

        if not image_path.exists():
            print(f"⚠️  No encontrada: {image_name}")
            error_count += 1
            continue

        try:
            # Cargar imagen
            img = Image.open(image_path).convert("RGB")

            # Aplicar transformación shrouded
            img_shrouded = apply_zoomout_fade(img)

            # Guardar en dataset_shrouded
            output_path = shrouded_dir / image_name
            img_shrouded.save(output_path, quality=95)

            processed_count += 1

            if (processed_count % 100) == 0:
                print(f"✓ Procesadas {processed_count}/{len(original_data)} imágenes...")

        except Exception as e:
            print(f"❌ Error procesando {image_name}: {str(e)}")
            error_count += 1

    print(f"\n" + "=" * 70)
    print(f"✅ PROCESO COMPLETADO")
    print(f"=" * 70)
    print(f"✓ Imágenes procesadas: {processed_count}")
    print(f"✓ Total guardadas en dataset_shrouded: {len(list(shrouded_dir.glob('*.png')))}")
    if error_count > 0:
        print(f"⚠️  Errores: {error_count}")
    print(f"=" * 70)

if __name__ == "__main__":
    load_and_process_images()

