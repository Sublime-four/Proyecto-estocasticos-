import json
import numpy as np
import scipy.signal as signal
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

METADATA_FILE = "recordings_metadata.json"
RESULTS_FILE = "audio_analysis.json"
GRAPH_FOLDER = "graphs"

# Crear carpeta para guardar gráficos
os.makedirs(GRAPH_FOLDER, exist_ok=True)

def cargar_metadata():
    """Carga los metadatos de las grabaciones."""
    try:
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def normalizar(valores):
    """Normaliza los valores usando min-max scaling."""
    min_val, max_val = np.min(valores), np.max(valores)
    return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0 for v in valores]

def calcular_psd(y, sr):
    """Calcula la densidad espectral de potencia (PSD) usando el método de Welch."""
    f, Pxx = signal.welch(y, fs=sr, nperseg=1024)
    return f, Pxx

def calcular_snr(y):
    """Calcula la relación señal-ruido (SNR)."""
    potencia_total = np.mean(y ** 2)
    ruido = y - np.mean(y)
    potencia_ruido = np.mean(ruido ** 2)
    return 10 * np.log10(potencia_total / potencia_ruido) if potencia_ruido > 0 else 0

def analizar_audio(ruta, tipo):
    """Calcula métricas avanzadas del audio."""
    y, sr = librosa.load(ruta, sr=44100)

    # Autocorrelación
    autocorr = signal.correlate(y, y, mode='full')[-len(y):]
    
    # Autocovarianza
    auto_cov = autocorr - np.mean(y) ** 2
    
    # Espectro de frecuencia (FFT)
    espectro = np.abs(np.fft.fft(y))

    # Densidad espectral de potencia (PSD)
    f, Pxx = calcular_psd(y, sr)

    # Curtosis y Skewness
    curtosis_val = kurtosis(y)
    skewness_val = skew(y)

    # SNR y rango dinámico
    snr_val = calcular_snr(y)
    rango_dinamico = np.max(y) - np.min(y)

    # Visualización del espectro y autocorrelación
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(f, Pxx)
    plt.title(f'PSD - {tipo}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Densidad espectral')
    
    plt.subplot(1, 2, 2)
    plt.plot(autocorr[:500])  # Solo los primeros valores para mejor visualización
    plt.title(f'Autocorrelación - {tipo}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelación')
    
    graph_path = os.path.join(GRAPH_FOLDER, f"{tipo}_{os.path.basename(ruta)}.png")
    plt.savefig(graph_path)
    plt.close()

    return {
        "autocorrelacion_media": float(np.mean(autocorr)),
        "autocovarianza_media": float(np.mean(auto_cov)),
        "espectro_media": float(np.mean(espectro)),
        "curtosis": float(curtosis_val),
        "skewness": float(skewness_val),
        "snr": float(snr_val),
        "rango_dinamico": float(rango_dinamico),
        "grafico": graph_path
    }

def analizar_todos():
    """Analiza todas las grabaciones y guarda los resultados."""
    metadata = cargar_metadata()
    resultados = {"cancion": [], "ruido_blanco": []}

    for dato in metadata:
        tipo, ruta = dato["tipo"], dato["ruta"]
        if os.path.exists(ruta):
            print(f"🔍 Analizando: {ruta}")
            analisis = analizar_audio(ruta, tipo)
            resultados[tipo].append(analisis)

    # Calcular valores medios y estándar con normalización
    resumen = {}
    for tipo, datos in resultados.items():
        if datos:
            resumen[tipo] = {
                "autocorrelacion_media": float(np.mean(normalizar([d["autocorrelacion_media"] for d in datos]))),
                "autocovarianza_media": float(np.mean(normalizar([d["autocovarianza_media"] for d in datos]))),
                "espectro_media": float(np.mean(normalizar([d["espectro_media"] for d in datos]))),
                "curtosis_media": float(np.mean([d["curtosis"] for d in datos])),
                "skewness_media": float(np.mean([d["skewness"] for d in datos])),
                "snr_media": float(np.mean([d["snr"] for d in datos])),
                "rango_dinamico_media": float(np.mean([d["rango_dinamico"] for d in datos]))
            }

    with open(RESULTS_FILE, "w") as f:
        json.dump(resumen, f, indent=4)

    print("📊 Análisis finalizado y guardado en", RESULTS_FILE)

if __name__ == "__main__":
    analizar_todos()
