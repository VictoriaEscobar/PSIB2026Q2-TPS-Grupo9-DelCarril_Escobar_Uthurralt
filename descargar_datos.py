"""
==========================================================================
DESCARGA DE DATOS 
==========================================================================
Descarga los archivos EEG del dataset PhysioNet Auditory EEG (v1.0.0).

Organiza los datos en carpetas según las cuatro condiciones experimentales:
reposo con ojos abiertos, reposo con ojos cerrados, música nativa y música
no nativa. Además, genera una copia plana de los archivos correspondientes
a los experimentos 5 y 6 para facilitar su uso en el preprocesamiento.

También descarga los archivos de metadatos (Subjects.csv, Songs.csv y
README.txt). Incluye reintentos ante errores de descarga y registra los
archivos que no pudieron descargarse.
"""


import os, time, shutil, requests

RAIZ = "https://physionet-open.s3.amazonaws.com/auditory-eeg/1.0.0/"
DATOS = "datos"

# Estructura para análisis música vs reposo: una carpeta por condición dentro de Segmented_Data_2.
DEST2 = os.path.join(DATOS, "Segmented_Data_2")

# Estructura original (plana, solo ex05 y ex06 para preprocesamiento y PSD.
DEST_PLANA = os.path.join(DATOS, "Segmented_Data")
COPIA_PLANA = True  # False = no generar datos/Segmented_Data/

# (carpeta, experimento, sesiones por sujeto)
#   ex01 / ex02: reposo, 3 sesiones por sujeto  -> sXX_ex0Y_sZZ.csv (120 archivos)
#   ex05 / ex06: música con auriculares comunes -> sXX_ex0Y.csv     (40 archivos)
CONDICIONES = [
    ("Sin música ojos abiertos", 1, 3),
    ("Sin música ojos cerrados", 2, 3),
    ("Con música nativa",        5, None),
    ("Con música no nativa",     6, None),
]

fallidos = []


def bajar(url, ruta, intentos=3):
    nombre = os.path.basename(ruta)
    if os.path.exists(ruta):
        print("ya estaba:", nombre, flush=True)
        return
    for i in range(1, intentos + 1):
        try:
            print(f"bajando {nombre} (intento {i})...", flush=True)
            with requests.get(url, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                with open(ruta + ".part", "wb") as f:
                    for trozo in r.iter_content(chunk_size=65536):
                        f.write(trozo)
            os.replace(ruta + ".part", ruta)
            print("  ok", flush=True)
            return
        except requests.RequestException as e:
            print("  falló:", e, flush=True)
            time.sleep(2)
    fallidos.append(nombre)


def nombres_experimento(ex, sesiones):
    """Nombres de archivo del dataset para un experimento."""
    for s in range(1, 21):
        if sesiones is None:
            yield f"s{s:02d}_ex{ex:02d}.csv"
        else:
            for ses in range(1, sesiones + 1):
                yield f"s{s:02d}_ex{ex:02d}_s{ses:02d}.csv"


os.makedirs(DATOS, exist_ok=True)
print("Destino:", os.path.abspath(DEST2), flush=True)

for carpeta, ex, sesiones in CONDICIONES:
    destino = os.path.join(DEST2, carpeta)
    os.makedirs(destino, exist_ok=True)
    print(f"\n== {carpeta} (ex{ex:02d}) ==", flush=True)
    for nombre in nombres_experimento(ex, sesiones):
        bajar(RAIZ + "Segmented_Data/" + nombre, os.path.join(destino, nombre))

# Copia plana ex05/ex06 para el código que ya existe (sin volver a descargar).
if COPIA_PLANA:
    os.makedirs(DEST_PLANA, exist_ok=True)
    for carpeta, ex, sesiones in CONDICIONES:
        if ex not in (5, 6):
            continue
        for nombre in nombres_experimento(ex, sesiones):
            origen = os.path.join(DEST2, carpeta, nombre)
            copia = os.path.join(DEST_PLANA, nombre)
            if os.path.exists(origen) and not os.path.exists(copia):
                shutil.copyfile(origen, copia)

print("\n== Metadatos ==", flush=True)
for nombre in ("Subjects.csv", "Songs.csv", "README.txt"):
    bajar(RAIZ + nombre, os.path.join(DATOS, nombre))

print("Listo." if not fallidos else f"Terminó, pero fallaron: {fallidos}")