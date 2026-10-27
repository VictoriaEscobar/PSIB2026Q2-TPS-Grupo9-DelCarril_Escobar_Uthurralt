# PSIB2026Q2-TPS-Grupo9-DelCarril_Escobar_Uthurralt
Trabajo Práctico Señales  - PSIB (16.63)

## Datos

Se usa el dataset **Auditory evoked potential EEG-Biometric dataset** (versión 1.0.0), de PhysioNet (acceso abierto):

Abo Alzahab N, Di Iorio A, Apollonio L, Alshalak M, Gravina A, Antognoli L, Baldi M, Scalise L, Alchalabi B. Auditory evoked potential EEG-Biometric dataset (version 1.0.0). PhysioNet. 2021. https://doi.org/10.13026/ps31-fc50

Del dataset se usan solo los registros de la carpeta `Segmented_Data` correspondientes a los experimentos 5 (canción en idioma nativo) y 6 (canción en idioma no nativo), ambos con auriculares comunes, para los 20 sujetos (40 archivos CSV). También se usan `Subjects.csv` y `Songs.csv`, que traen los metadatos de los sujetos y las canciones.

Tras la evaluación de calidad, se excluyeron del presente estudios los registros "s19_ex05" y "s19_ex06" por una ambigüedad en los metadatos del sujeto respecto a su lengua materna.

Los datos **no se incluyen en el repositorio**. Se descargan con el script `descargar_datos.py`, que los baja desde la copia pública de PhysioNet en Amazon S3 y los guarda en la carpeta `datos/`.

## Cómo reproducir

1. Clonar el repositorio y, parado en su carpeta, crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1      # Windows (PowerShell)
   source .venv/bin/activate       # Mac / Linux / WSL
   ```

2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Descargar los datos (unos 76 MB):

   ```bash
   python descargar_datos.py
   ```

Al terminar, `datos/Segmented_Data/` debe tener 40 archivos CSV.

