
"""
==========================================================================
ANÁLISIS ESPECTRAL Y CÁLCULO DE POTENCIA POR BANDAS
==========================================================================
Procesa señales EEG para obtener su densidad espectral de potencia (PSD)
mediante una implementación manual del método de Welch y calcular la
potencia absoluta y relativa en distintas bandas de frecuencia.

Incluye funciones para:
- Integrar señales mediante la regla trapezoidal, adaptándose a distintas
  versiones de NumPy.
- Calcular la PSD de una señal utilizando el método de Welch, con ventanas
  Hann y segmentos solapados.
- Aplicar el cálculo espectral individualmente a los canales de un
  DataFrame de EEG.
- Calcular la potencia absoluta y relativa de cada banda de frecuencia
  mediante la integración de la PSD.
- Obtener la potencia relativa respecto de un rango total de frecuencias
  definido como referencia.

El procesamiento permite caracterizar la distribución de potencia de las
señales EEG en diferentes bandas de frecuencia, facilitando su posterior
análisis y comparación entre registros o condiciones experimentales.
==========================================================================
"""


import numpy as np
import scipy.signal

def integrar(y, x):
    """Integra usando trapz o trapezoid según la versión de numpy disponible."""
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x)
    return np.trapz(y, x)


def welch_per(x, fs, nperseg, nfft, noverlap):
    """
    Calcula la densidad espectral de potencia (PSD) mediante
    el método de Welch con ventana Hann y segmentos solapados.
    """
    x = np.asarray(x, dtype=float)

    if nperseg <= 0 or nfft < nperseg:
        raise ValueError("nfft debe ser mayor o igual a nperseg.")

    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError("noverlap debe ser menor que nperseg.")

    if len(x) < nperseg:
        raise ValueError("La señal es más corta que nperseg.")

    step = nperseg - noverlap

    win = scipy.signal.windows.hann(nperseg, sym=False)

    escala = fs * np.sum(win**2)

    psd = np.zeros(nfft // 2 + 1)
    count = 0

    for start in range(0, len(x) - nperseg + 1, step):

        segment = x[start:start + nperseg]

        # Eliminación de la media de cada segmento
        segment = segment - np.mean(segment)

        windowed = segment * win

        fft_segment = np.fft.rfft(windowed, n=nfft)

        psd_segment = np.abs(fft_segment)**2 / escala

        # Conversión a espectro unilateral
        if nfft % 2 == 0:
            psd_segment[1:-1] *= 2
        else:
            psd_segment[1:] *= 2

        psd += psd_segment
        count += 1

    psd /= count

    freq = np.fft.rfftfreq(nfft, d=1/fs)

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

def potencia_por_banda(freq, psd, bandas, f_total=(0.5, 40)):
    """
    Integra la PSD dentro de cada banda de frecuencia (regla trapezoidal)
    y calcula la potencia relativa respecto al rango f_total.

    Parámetros:
    freq: vector de frecuencias (Hz), de welch_per
    psd: vector de PSD correspondiente
    bandas: diccionario {nombre_banda: (f_min, f_max)}
    f_total: tupla (f_min, f_max) sobre la que se calcula la potencia total
             de referencia para la relativa (default: rango del filtro pasabanda)

    Retorna:
    dict {f"{nombre_banda}_abs": potencia_absoluta, f"{nombre_banda}_rel": potencia_relativa}

    Lanza:
    ValueError si alguna banda se sale del rango f_total.
    """
    # Chequeo: todas las bandas deben estar dentro de f_total
    for nombre, (f_min, f_max) in bandas.items():
        if f_min < f_total[0] or f_max > f_total[1]:
            raise ValueError(
                f"La banda '{nombre}' ({f_min}-{f_max} Hz) se sale de f_total {f_total}."
            )

    mascara_total = (freq >= f_total[0]) & (freq <= f_total[1])
    potencia_total = integrar(psd[mascara_total], freq[mascara_total])

    resultado = {}
    for nombre, (f_min, f_max) in bandas.items():
        mascara = (freq >= f_min) & (freq < f_max)   # < en vez de <=
        pot_abs = integrar(psd[mascara], freq[mascara])
        resultado[f"{nombre}_abs"] = pot_abs
        resultado[f"{nombre}_rel"] = pot_abs / potencia_total
    return resultado