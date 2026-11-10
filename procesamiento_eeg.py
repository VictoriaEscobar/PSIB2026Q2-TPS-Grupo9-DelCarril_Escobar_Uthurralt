"""
==========================================================================
ANÁLISIS DE CALIDAD Y FILTRADO
==========================================================================
Trabaja con los registros de los experimentos descargados del dataset
Auditory evoked potential EEG-Biometric (PhysioNet v1.0.0).

Incluye funciones para:
- Cargar registros EEG desde archivos CSV.
- Verificar la calidad e integridad de los registros.
- Auditar el dataset y generar reportes de calidad.
- Detectar canales sospechosos mediante métricas estadísticas.
- Visualizar señales en el dominio temporal y su densidad espectral de potencia (PSD).
- Comparar registros crudos y filtrados.
- Aplicar filtros Notch de 50 Hz y pasabandas Butterworth de 0.1 a 40 Hz.
"""
# ========================  IMPORTS  ===========================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from scipy import signal
from scipy.signal import welch


# ======================  CONSTANTES  ===========================

fs = 200               # Frecuencia de muestreo (Hz)
min_muestras = 18000   # 90 s a 200 Hz: mínimo para considerar válido un registro (de 120 s)

# =================== CARGA DE REGISTROS =========================

def cargar_registro(ruta_archivo: str) -> pd.DataFrame:
    """
    Lee un CSV de EEG.

    - La primera columna del CSV es el índice de muestra: pasa a ser el índice
      del DataFrame y NO se cuenta como canal.
    - Se conservan las columnas de canales presentes en el archivo.
    - Todo se convierte a número, lo que no se pueda convertir queda como NaN.

    """
    df = pd.read_csv(ruta_archivo, index_col=0)
    df.columns = df.columns.str.strip()
    return df.apply(pd.to_numeric, errors="coerce")



# =================== CONTROL DE CALIDAD ========================


def verificar_calidad_csv(ruta_archivo: str, min_muestras: int):
    """
    Verifica la calidad e integridad de un registro EEG individual.

    Comprueba que el archivo no esté vacío, que contenga canales de EEG,
    que no haya valores nulos o infinitos y que tenga la cantidad mínima
    de muestras requerida.

    Parámetros:
        ruta_archivo: Ruta del archivo CSV a evaluar.
        min_muestras: Cantidad mínima de muestras requeridas.

    Devuelve:
        Una tupla (valido, motivo), indicando si el registro es válido
        y, en caso contrario, el motivo del rechazo.
    """
    try:
        df = cargar_registro(ruta_archivo) # Lectura del archivo.

        # Prueba 1: verifica que el archivo no esté vacío.
        if df.empty:
            return False, "El archivo está vacío"

        canales = df.columns

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
            
        # Validación de calidad del registro utilizando todos los canales disponibles.
        valido, motivo = verificar_calidad_csv(ruta_completa, min_muestras)

        # Extraer metadatos
        partes = archivo.replace('.csv', '').split('_')
        sujeto = partes[0]
        condicion = 'Nativo' if '_ex05' in archivo else 'No_Nativo'

        # Guardamos
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
    Calcula métricas estadísticas por canal para identificar señales
    potencialmente defectuosas o sospechosas.

    Evalúa el desvío estándar, la proporción de diferencias consecutivas
    iguales a cero y la amplitud máxima de cada canal. Compara los
    resultados con la mediana de los canales del mismo registro.

    Devuelve:
        Un DataFrame con las métricas calculadas y la identificación
        de los canales sospechosos.
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



# ============================================================
# EXPLORACIÓN Y VISUALIZACIÓN DE REGISTROS EEG
# ============================================================
#
# Permite visualizar las señales EEG en el dominio temporal
# y su densidad espectral de potencia (PSD) mediante Welch.
#
# Uso:
#
# 1) Visualización de registros crudos:
# explorar_registros(
#     {"crudo": ruta_datos},
#     archivos, n_muestras=2, seed=0, fs=fs
# )
#
# 2) Comparación de registros crudos y filtrados:
# explorar_registros(
#     {"crudo": ruta_datos, "filtrado": ruta_filtrados},
#     archivos, n_muestras=2, seed=0, fs=fs
# )
#
# 3) Comparación de señales filtradas en memoria:
# df_crudo = cargar_registro(os.path.join(ruta_datos, archivo))
# df_filt = aplicar_filtros_eeg(df_crudo, fs)
# comparar_senales(
#     {"crudo": df_crudo, "filtrado": df_filt},
#     fs=fs, titulo=archivo
# )

COLORES = ["steelblue", "darkorange", "seagreen", "crimson"]

# {frecuencia_Hz: (color, estilo)}
REFS_DEFAULT = {
    0.1: ("green", ":"),
    40: ("orange", ":"),
    50: ("red", "--"),
}

def calcular_psd(x, fs, nperseg):
    """PSD de Welch ignorando NaN. Devuelve (None, None) si no hay datos."""
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return None, None
    f, psd = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    return f, np.maximum(psd, np.finfo(float).tiny)  # evita ceros en escala log


def comparar_senales(
    senales,
    fs,
    seg_tiempo=10,
    inicio=0,
    nperseg=4000,
    fmax=60,
    refs_hz=REFS_DEFAULT,
    titulo="",
):
    """
    Grafica, por canal, la serie temporal (columna izquierda) y la PSD (columna derecha)
    de una o más versiones de un mismo registro.

    Parámetros
    ----------
    senales : dict {nombre: DataFrame}
        Ej.: {"crudo": df_crudo, "filtrado": df_filt}. Todos con los mismos canales.
    fs : float
        Frecuencia de muestreo (Hz).
    seg_tiempo, inicio : float
        Duración y comienzo (s) del fragmento temporal a mostrar.
    nperseg : int
        Muestras por segmento de Welch (se limita al largo del registro).
    fmax : float
        Frecuencia máxima de la PSD.
    refs_hz : dict o None
        Líneas verticales de referencia en la PSD.
    """
    canales = list(next(iter(senales.values())).columns)
    for nombre, df in senales.items():
        if list(df.columns) != canales:
            raise ValueError(f"'{nombre}' tiene canales distintos: {list(df.columns)}")

    i0 = int(inicio * fs)
    i1 = i0 + int(seg_tiempo * fs)

    fig, axes = plt.subplots(
        len(canales), 2, figsize=(15, 2.8 * len(canales)), squeeze=False
    )

    for i, canal in enumerate(canales):
        ax_t, ax_f = axes[i]

        for (nombre, df), color in zip(senales.items(), COLORES):
            x = df[canal].to_numpy(dtype=float)

            seg = x[i0:i1]
            t = (i0 + np.arange(len(seg))) / fs
            ax_t.plot(t, seg, color=color, lw=0.8, alpha=0.85, label=nombre)

            f, psd = calcular_psd(x, fs, nperseg)
            if f is not None:
                ax_f.semilogy(f, psd, color=color, lw=1, alpha=0.9, label=nombre)

        ax_t.set_ylabel(canal)
        ax_t.grid(True, ls="--", alpha=0.4)

        for f_ref, (c, ls) in (refs_hz or {}).items():
            ax_f.axvline(f_ref, color=c, ls=ls, alpha=0.8, label=f"{f_ref} Hz")
        ax_f.set_xlim(0, min(fmax, fs / 2))
        ax_f.set_ylabel("PSD (unidad²/Hz)")
        ax_f.grid(True, which="both", ls="--", alpha=0.4)

        if i == 0:
            ax_t.set_title(f"Serie temporal ({seg_tiempo} s desde t={inicio} s)")
            ax_f.set_title("PSD de Welch (registro completo)")
            ax_t.legend(fontsize=8)
            ax_f.legend(fontsize=8)
        if i == len(canales) - 1:
            ax_t.set_xlabel("Tiempo (s)")
            ax_f.set_xlabel("Frecuencia (Hz)")

    n_reg = len(next(iter(senales.values())))
    fig.suptitle(f"{titulo} — {n_reg / fs:.1f} s, resolución Welch "
                 f"{fs / min(nperseg, n_reg):.3f} Hz", fontsize=13)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def explorar_registros(
    rutas,
    lista_archivos,
    n_muestras=2,
    seed=None,
    **kwargs,
):
    """
    Elige archivos al azar, los carga de cada carpeta en `rutas` y los compara.

    Parámetros
    ----------
    rutas : dict {nombre: carpeta}
        Ej.: {"crudo": "datos/raw"} o {"crudo": "datos/raw", "filtrado": "datos/filt"}.
        Los archivos deben llamarse igual en todas las carpetas.
    lista_archivos : list
        Nombres de archivo candidatos.
    n_muestras : int
        Cuántos archivos elegir al azar.
    seed : int o None
        Semilla para reproducir la selección.
    **kwargs
        Se pasan a `comparar_senales` (fs, seg_tiempo, inicio, nperseg, fmax, refs_hz).

    Devuelve la lista de archivos seleccionados.
    """
    validos = [
        a for a in lista_archivos
        if a.lower().endswith(".csv")
        and all(os.path.isfile(os.path.join(r, a)) for r in rutas.values())
    ]
    if not validos:
        raise ValueError("No hay CSV presentes en todas las carpetas indicadas.")

    sel = random.Random(seed).sample(validos, k=min(n_muestras, len(validos)))

    for arch in sel:
        senales = {n: cargar_registro(os.path.join(r, arch)) for n, r in rutas.items()}
        comparar_senales(senales, titulo=arch, **kwargs)

    return sel

# ============================================================
# FILTRADO
# ============================================================

def aplicar_filtros_eeg(df_eeg, fs, fc_baja=0.1, fc_alta=40.0, f_notch=50.0, q_notch=30.0, order=4):
    """
    Aplica un filtrado Notch y un filtro pasabandas Butterworth a los
    canales EEG disponibles.

    Elimina la interferencia de 50 Hz mediante el filtro Notch y
    limita el contenido frecuencial al rango de 0.1 a 40 Hz mediante
    el filtro pasabandas. Ambos filtros se aplican en fase cero.

    Parámetros:
        df_eeg: DataFrame con los registros EEG.
        fs: Frecuencia de muestreo en Hz.
        fc_baja: Frecuencia de corte inferior del pasabandas.
        fc_alta: Frecuencia de corte superior del pasabandas.
        f_notch: Frecuencia central del filtro Notch.
        q_notch: Factor de calidad del filtro Notch.
        order: Orden del filtro Butterworth.

    Devuelve:
        Un DataFrame con las señales filtradas.
    """
    df_filtrado = df_eeg.copy()
    
    # Canales 
    canales_objetivo = ['T7', 'F8', 'Cz', 'P4']
    canales = [c for c in canales_objetivo if c in df_filtrado.columns]

    # 1. Diseño del filtro Notch (50 Hz)
    b_notch, a_notch = signal.iirnotch(w0=f_notch, Q=q_notch, fs=fs)

   # 2. Diseño del pasabandas en forma SOS (más estable)
    sos_band = signal.butter(N=order, Wn=[fc_baja, fc_alta], btype='bandpass', fs=fs, output='sos')
    # 3. Filtrado canal por canal con fase cero (filtfilt, forward-backward)

    for canal in canales:
        x = df_filtrado[canal].values.astype(float)
        
        # Filtrado en cascada hacia adelante y hacia atrás
        x_notch = signal.filtfilt(b_notch, a_notch, x)   
        x_filt = signal.sosfiltfilt(sos_band, x_notch)
        
        df_filtrado[canal] = x_filt

    return df_filtrado