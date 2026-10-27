"""
Procesamiento de EEG para el TPI de PSIB (Grupo 9).
 
Dataset: Auditory evoked potential EEG-Biometric dataset v1.0.0 (PhysioNet).
Se usan los registros de Segmented_Data de los experimentos 5 (canción en
idioma nativo) y 6 (canción en idioma no nativo), auriculares comunes.
 
Flujo: cargar datos (ejecutar descargar_datos.py) -> (completar)
"""

# ========================  IMPORTS  ===========================
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os



# =================== CONTROL DE CALIDAD ========================


def verificar_calidad_csv(ruta_archivo: str, min_muestras: int):
    """
    Verifica la integridad de un DataFrame de EEG individual.
    - Retorna (True, None) si el archivo es válido (lugares no vacíos, con valores válidos).
    - Retorna (False, "motivo") si el archivo es inválido.

    Se le debe pasar:
    - Ruta del archivo del dataset a evaluar (ruta_archivo).
    - Cantidad mínima de filas (muestras) que debe tener la señal para ser considerada válida.
    Para 2 minutos de registro a 200 Hz, se recomienda min_muestras = 18000 (1 min 30 segundos de señal.)
    """
    try:
        df = pd.read_csv(ruta_archivo) # Lectura del archivo.

        # Prueba 1: verifica que el archivo no esté vacío.
        if df.empty:
            return False, "El archivo está vacío"

        canales = df.columns[1:]

        # Prueba 2: verifica que se hacen las columnas de registro de canales.
        if len(canales)==0:
            return False, "No se hallaron las columnas de canales de EEG"

        # Prueba 3: verifica si las columnas poseen valores Nan o Null en las columnas.
        if df[canales].isnull().any().any():
            return False, "Contiene valores nulos o faltantes (NaN/Null)"

        # Prueba 4: verifica si las columnas (canales) poseen valores infinitos.
        for canal in canales:
            if pd.api.types.is_numeric_dtype(df[canal]):
                if np.isinf(df[canal]).any():
                    return False, f"El canal '{canal}' contiene valores infinitos"

        # Prueba 5: verifica que el archivo tenga una cantidad de muestras apropiada
        # y en concordancia con fs y el tiempo de registro.
        if len(df) < min_muestras:
            return False, f"Señal demasiado corta ({len(df)} muestras)"

        # Si pasa todos los chequeos, archivo válido.
        return True, None

    # Excepción si no logra leer el .CSV.
    except Exception as e:
        return False, f"Error de lectura en el CSV: {str(e)}"




def auditar_dataset_segmentado(ruta_Segmented_Data):
    """
    Recorre los archivos "ex05" (nativo) y "ex06" (no nativo) en la carpeta Segmented_Data,
    ejecuta la verificación de verificar_calidad_csv sobre cada uno evaluando sus canales y genera un reporte.
    """
    reporte = []

    if not os.path.exists(ruta_Segmented_Data):
        print(f"ERROR: La ruta '{ruta_Segmented_Data}' no existe.")
        return pd.DataFrame(columns=['archivo', 'sujeto', 'condicion', 'valido', 'observacion'])
    
    # Listar todos los archivos .csv en la carpeta
    archivos = [f for f in os.listdir(ruta_Segmented_Data) if f.endswith('.csv')]
        
    # Filtrar únicamente Experimento 5 (Nativo) y Experimento 6 (No Nativo)
    archivos_interes = [f for f in archivos if '_ex05' in f or '_ex06' in f]

    print(f"--- Auditoría de calidad sobre {len(archivos_interes)} archivos ---")

    # PREVENCIÓN: Si no hay archivos, retorna un DataFrame vacío sin romper
    if len(archivos_interes) == 0:
        print("No se encontraron archivos que coincidan con '_ex05_' o '_ex06_'")
        return pd.DataFrame(columns=['archivo', 'sujeto', 'condicion', 'valido', 'observacion'])

    for archivo in archivos_interes:
        ruta_completa = os.path.join(ruta_Segmented_Data, archivo)
            
        # Validación leyendo el CSV y revisando los canales desde df.columns[1:]
        valido, motivo = verificar_calidad_csv(ruta_completa, 18000)

        # Extraer metadatos
        partes = archivo.replace('.csv', '').split('_')
        sujeto = partes[0]
        condicion = 'Nativo' if '_ex05' in archivo else 'No_Nativo'

        # Guardamos SIN acentos en las llaves para evitar KeyErrors
        reporte.append({
            'archivo': archivo,
            'sujeto': sujeto,
            'condicion': condicion,
            'valido': valido,
            'observacion': 'OK' if valido else motivo
        })

    df_reporte = pd.DataFrame(reporte)
        
    # Resumen por consola
    total = len(df_reporte)
    validos = df_reporte['valido'].sum()
    print(f"\nResumen del Control de Calidad:")
    print(f"Archivos válidos: {validos} / {total}")
        
    if validos < total:
        print(f"Archivos rechazados ({total - validos}):")
        print(df_reporte[df_reporte['valido'] == False][['archivo', 'observacion']])
    else:
        print("Todos los archivos pasaron el control de calidad perfectamente")

    return df_reporte



def screening_canales(ruta_Segmented_Data, archivos, factor_bajo=0.1, factor_alto=5.0):
    """
    Calcula métricas estadísticas por canal y marca canales sospechosos comparando
    contra la mediana de los canales del mismo registro.
    """
    filas = []
    for archivo in archivos:
        df = pd.read_csv(os.path.join(ruta_Segmented_Data, archivo))
        canales = df.columns[1:]
        x = df[canales].to_numpy(dtype=float) # Sus filas son muestras, sus columnas son canales. 1 matriz por cada archivo.

        desvio = x.std(axis=0) # Un canal muerto tiene desvío casi 0, y uno con artefactos grandes tiene desvío muy alto.
        frac_cero = (np.diff(x, axis=0) == 0).mean(axis=0) # Proporción de pasos en los que la señal no cambió entre una muestra y la siguiente (cerca de 0 es normal, cerca de 1 es señal congelada).
        amp_max = np.abs(x - x.mean(axis=0)).max(axis=0) # Amplitud máxima quitando el offset de continua  y tomando valor absoluto.
        std_ref = np.median(desvio) # Referencia: mediana de los desvíos de todos los canales de ese registro.


        for i, canal in enumerate(canales):
            motivo = []
            if desvio[i] < factor_bajo * std_ref: # Desvio < 0,1 x mediana ==> señal casi plana.
                motivo.append("Desvío estándar muy bajo") 
            if desvio[i] > factor_alto * std_ref: # Desvio > 5 x mediana ==> mucho ruido.
                motivo.append("Desvío estándar muy alto")
            if frac_cero[i] > 0.5:  # Mas de la mitad de diferencias iguales a cero ==> señal congelada.
                motivo.append("Señal congelada")
                
            filas.append({
                'Archivo': archivo, 'Canal': canal,
                'Desvío estándar': desvio[i], 'Proporción de diferencias entre muestras iguales a cero': frac_cero[i],
                'Amplitud máxima': amp_max[i],
                'Sospechoso': len(motivo) > 0,
                'Motivo': ", ".join(motivo)
            })
    return pd.DataFrame(filas)