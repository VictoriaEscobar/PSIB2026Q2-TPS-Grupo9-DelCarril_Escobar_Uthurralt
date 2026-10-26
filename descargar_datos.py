import os, time, requests

RAIZ = "https://physionet-open.s3.amazonaws.com/auditory-eeg/1.0.0/"
DATOS = "datos"
DEST = os.path.join(DATOS, "Segmented_Data")
os.makedirs(DEST, exist_ok=True)
print("Destino:", os.path.abspath(DEST), flush=True)

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

for s in range(1, 21):
    for ex in (5, 6):
        nombre = f"s{s:02d}_ex{ex:02d}.csv"
        bajar(RAIZ + "Segmented_Data/" + nombre, os.path.join(DEST, nombre))

for nombre in ("Subjects.csv", "Songs.csv", "README.txt"):
    bajar(RAIZ + nombre, os.path.join(DATOS, nombre))

print("Listo." if not fallidos else f"Terminó, pero fallaron: {fallidos}")