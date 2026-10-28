import numpy as np
import scipy.signal

def welch_per(x, fs, nperseg, nfft, noverlap):
    """
    Implementa el cálculo del espectro de densidad de potencia (PSD) usando el método de Welch
    con segmentos solapados (overlapping).

    Parámetros:
    x: señal
    fs: frecuencia de muestreo en Hz
    nperseg: cantidad de muestras por segmento (largo de ventana)
    nfft: número de puntos de la FFT
    noverlap: cantidad de muestras que se superponen

    window: utiliza hanning
    """
    step = int(nperseg - noverlap)
    win = scipy.signal.windows.hann(nperseg)

    psd = np.zeros(nfft)
    count = 0

    for start in range(0, len(x) - nperseg + 1, step):
        segment = x[start:start + nperseg]
        windowed = segment * win
        fft_segment = np.fft.fft(windowed, n=nfft)
        psd += np.abs(fft_segment)**2 / nfft
        count += 1

    psd /= count
    psd = psd[:nfft // 2] * 2
    freq = np.linspace(0, fs / 2, nfft // 2)

    return freq, psd

def welch_eeg(df, fs, nperseg=4000, nfft=None, noverlap=None, canales=None):
    """
    Aplica welch_per (Welch manual) a cada canal de un DataFrame de EEG.

    Parámetros:
    df: DataFrame con las señales EEG (una columna por canal: T7, F8, Cz, P4)
    fs: frecuencia de muestreo en Hz
    nperseg: muestras por segmento (default: 4000, igual que en comparar_senales)
    nfft: puntos de la FFT (default: igual a nperseg)
    noverlap: muestras de solapamiento (default: 50% de nperseg)
    canales: lista de columnas a procesar (default: todas las del df)

    Retorna:
    freq: vector de frecuencias (Hz)
    psd_dict: diccionario {canal: psd}
    """
    if canales is None:
        canales = list(df.columns)

    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nperseg // 2

    psd_dict = {}
    freq = None

    for canal in canales:
        x = df[canal].to_numpy(dtype=float)
        f, psd = welch_per(x, fs, nperseg, nfft, noverlap)
        psd_dict[canal] = psd
        if freq is None:
            freq = f

    return freq, psd_dict
import numpy as np


def potencia_por_banda(freq, psd, bandas=BANDAS):
    """
    Integra la PSD dentro de cada banda de frecuencia (regla trapezoidal).

    Parámetros:
    freq: vector de frecuencias (Hz), de welch_per
    psd: vector de PSD correspondiente
    bandas: diccionario {nombre_banda: (f_min, f_max)}

    Retorna:
    dict {nombre_banda: potencia}
    """
    resultado = {}
    for nombre, (f_min, f_max) in bandas.items():
        mascara = (freq >= f_min) & (freq <= f_max)
        resultado[nombre] = np.trapz(psd[mascara], freq[mascara])
    return resultado