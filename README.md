# PSIB2026Q2-TPS-Grupo9-DelCarril_Escobar_Uthurralt

Trabajo Práctico Señales - PSIB (16.63)

Análisis comparativo de señales EEG de escucha de música (canción en idioma nativo y no nativo) y de reposo (ojos abiertos y ojos cerrados) y, a partir de un dataset público de PhysioNet.

## Datos

Se usa el dataset **Auditory evoked potential EEG-Biometric dataset** (versión 1.0.0), de PhysioNet (acceso abierto):

Abo Alzahab N, Di Iorio A, Apollonio L, Alshalak M, Gravina A, Antognoli L, Baldi M, Scalise L, Alchalabi B. Auditory evoked potential EEG-Biometric dataset (version 1.0.0). PhysioNet. 2021. [https://doi.org/10.13026/ps31-fc50](https://doi.org/10.13026/ps31-fc50)

El dataset contiene registros de 20 voluntarios, con 4 canales (T7, F8, Cz y P4) a 200 Hz. De la carpeta `Segmented_Data` (2 minutos de señal con el menor ruido de movimiento posible por registro) se usan cuatro condiciones:

| Condición | Experimento | Descripción | Archivos |
|---|---|---|---|
| Sin música, ojos abiertos | ex01 | Reposo con ojos abiertos, 3 sesiones por sujeto | 60 |
| Sin música, ojos cerrados | ex02 | Reposo con ojos cerrados, 3 sesiones por sujeto | 60 |
| Con música nativa | ex05 | Canción en idioma nativo, auriculares comunes | 20 |
| Con música no nativa | ex06 | Canción en idioma no nativo, auriculares comunes | 20 |

En total son 160 archivos CSV. También se usan `Subjects.csv` y `Songs.csv`, que traen los metadatos de los sujetos y las canciones.

Los nombres de archivo siguen el formato `sXX_exXX_sXX.csv` (sujeto, experimento y, solo en ex01 y ex02, sesión). Por ejemplo, `s03_ex02_s01.csv` es el sujeto 3, experimento 2, sesión 1. Los de música no llevan sesión (`s03_ex05.csv`).

Los experimentos 3 y 4 (reposo con cancelación de sonido) no están incluidos en el dataset, y los experimentos 7 a 10 (música neutra y conducción ósea) no se usan en este trabajo.

Tras la evaluación de calidad, se excluyeron del presente estudio los registros `s19_ex05` y `s19_ex06` por una ambigüedad en los metadatos del sujeto respecto a su lengua materna.

Los datos **no se incluyen en el repositorio**. Se descargan con el script `descargar_datos.py`, que los baja desde la copia pública de PhysioNet en Amazon S3 y los guarda en la carpeta `datos/` (ignorada por git).

## Instalación

Se necesita **Python 3.11 o superior** y `git`.

1. Clonar el repositorio y entrar a su carpeta:

   ```
   git clone https://github.com/VictoriaEscobar/PSIB2026Q2-TPS-Grupo9-DelCarril_Escobar_Uthurralt.git
   cd PSIB2026Q2-TPS-Grupo9-DelCarril_Escobar_Uthurralt
   ```

2. Crear y activar un entorno virtual:

   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1      # Windows (PowerShell)
   source .venv/bin/activate       # Mac / Linux / WSL
   ```

3. Instalar las dependencias:

   ```
   pip install -r requirements.txt
   ```

4. Descargar los datos (ver [Modo de uso](#modo-de-uso)).

## Requirements

Las dependencias están fijadas en `requirements.txt`. Las librerías que el proyecto usa directamente son:

| Librería | Uso |
|---|---|
| `numpy` | Cálculo numérico |
| `pandas` | Lectura y manejo de los registros CSV |
| `scipy` | Filtrado y estimación espectral (Welch) |
| `matplotlib` | Gráficos |
| `requests` | Descarga de los datos |
| `ipykernel` | Ejecución de los notebooks |

El resto de los paquetes de `requirements.txt` son dependencias de las anteriores.

## Modo de uso

### 1. Descargar los datos

```
python descargar_datos.py
```

El script es reanudable: si se corta, se vuelve a correr y salta los archivos que ya están. Al terminar, la carpeta `datos/` queda con la siguiente estructura:

```
datos/
├── Segmented_Data_2/
│   ├── Sin música ojos abiertos/    60 archivos (ex01)
│   ├── Sin música ojos cerrados/    60 archivos (ex02)
│   ├── Con música nativa/           20 archivos (ex05)
│   └── Con música no nativa/        20 archivos (ex06)
├── Segmented_Data/                  40 archivos (ex05 y ex06, copia plana)
├── Subjects.csv
├── Songs.csv
└── README.txt
```

`Segmented_Data/` es una copia de los experimentos 5 y 6 que se mantiene por compatibilidad con el código del análisis de música. Si no se necesita, se puede desactivar con `COPIA_PLANA = False` en `descargar_datos.py`.

### 2. Ejecutar el análisis

Los notebooks de la carpeta `notebooks/` se ejecutan en orden:

| Notebook | Contenido |
|---|---|
| `01_calidad_datos.ipynb` | Calidad de los datos |
| `02_filtrado.ipynb` | Filtrado |
| `03_Analisis_PSD.ipynb` | Análisis de la densidad espectral de potencia (PSD) |
| `04_descriptores.ipynb` | Descriptores de PSD en escucha de música nativa vs no nativa |
| `05_estadistica_y_exploratorio.ipynb` | Estadística y análisis exploratorio de escucha de música nativa vs no nativa|
| `06_pipeline_reposo_vs_musica.ipynb` | Pipeline completo de reposo vs. música |

Se pueden abrir desde VS Code o con Jupyter, parados en la raíz del repositorio y con el entorno virtual activado.

### 3. Módulos de código

Los notebooks usan las funciones de dos módulos que están en la raíz del repositorio:

- `procesamiento_eeg.py`: carga de registros, control de calidad, screening de canales, filtrado, PSD con Welch y gráficos comparativos.
- `analisis_PSD.py`: PSD de Welch implementada manualmente (`welch_per`, `welch_eeg`) y potencia por banda de frecuencia (`potencia_por_banda`).

Ejemplo de uso:

```python
from procesamiento_eeg import cargar_registro, fs
from analisis_PSD import welch_eeg

df = cargar_registro("datos/Segmented_Data_2/Con música nativa/s01_ex05.csv")
freq, psd = welch_eeg(df, fs)
```

## Estructura del repositorio

```
├── datos/                                   Datos descargados (no se suben al repo, se descargan con ´descargar_datos.py´)
│     ├── Segmented_Data_2/                  Datos crudos segmentados, por condición
│     │   ├── Sin música ojos abiertos/      
│     │   ├── Sin música ojos cerrados/      
│     │   ├── Con música nativa/            
│     │   └── Con música no nativa/          
│     ├── Segmented_Data/                   
│     ├── Subjects.csv                       Metadatos y cuestionario de los sujetos
│     ├── Songs.csv                          Canciones usadas
│     └── README.txt                         README original del dataset
├── notebooks/                               Notebooks de visualización y análisis
│     ├── 01_calidad_datos.ipynb             
│     ├── 02_filtrado.ipynb                 
│     ├── 03_Analisis_PSD.ipynb     
│     ├── 04_descriptores.ipynb   
│     ├── 05_estadistica_y_exploratorio.ipynb     
│     └── 06_pipeline_reposo_vs_musica.ipynb         
├── resultados/                              Resultados de la comparación música nativa vs. no nativa
├── resultados_reposo_musica/                Resultados de la comparación reposo vs. música
├── informe/                                 Informe del trabajo
├── descargar_datos.py                       Descarga de los archivos necesarios del dataset original
├── procesamiento_eeg.py                     Funciones de preprocesamiento: calidad y filtrado
├── analisis_PSD.py                          Funciones de PSD y potencia por banda
└── requirements.txt                         Dependencias
```