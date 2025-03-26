import json
import numpy as np
import pyaudio
import scipy.signal as signal
import librosa
import time
from scipy.stats import kurtosis, skew, entropy

RESULTS_FILE = "audio_analysis.json"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
DURATION = 2  # Duración de la captura en segundos

def cargar_umbrales():
    """Carga los valores medios de análisis desde audio_analysis.json."""
    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ No se encontró el archivo de análisis. Usando valores predeterminados.")
        return {}

def capturar_audio():
    """Captura audio en vivo y devuelve la señal normalizada como un array numpy."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("🎤 Capturando audio...")
    frames = [np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16) for _ in range(0, int(RATE / CHUNK * DURATION))]
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    y = np.concatenate(frames).astype(np.float32)  # Convertimos a float32
    y /= np.max(np.abs(y))  # Normalizamos entre -1 y 1
    return y

def calcular_entropia_espectral(y, sr=RATE):
    """Calcula la entropía espectral manualmente usando scipy.stats.entropy()."""
    espectro = np.abs(librosa.stft(y, n_fft=1024))  # Calculamos el STFT
    espectro = np.mean(espectro, axis=1)  # Promediamos sobre el tiempo
    espectro /= np.sum(espectro)  # Normalizamos para que sea una distribución de probabilidad
    entropia = entropy(espectro)  # Calculamos la entropía con scipy
    return entropia

def calcular_snr(y):
    """Calcula la relación señal-ruido (SNR)."""
    potencia_total = np.mean(y ** 2)
    ruido = y - np.mean(y)
    potencia_ruido = np.mean(ruido ** 2)
    return 10 * np.log10(potencia_total / potencia_ruido) if potencia_ruido > 0 else -50  # Limitamos SNR a -50dB si es muy bajo

def analizar_audio(y):
    """Calcula características avanzadas del audio en vivo."""
    autocorr = signal.correlate(y, y, mode='full')[-len(y):]
    auto_cov = autocorr - np.mean(y) ** 2

    entropia_espectral = calcular_entropia_espectral(y)
    curtosis_val = kurtosis(y)
    skewness_val = skew(y)
    snr_val = calcular_snr(y)
    rango_dinamico = np.max(y) - np.min(y)

    return {
        "autocorrelacion": float(np.mean(autocorr)),
        "autocovarianza": float(np.mean(auto_cov)),
        "entropia_espectral": float(entropia_espectral),
        "curtosis": float(curtosis_val),
        "skewness": float(skewness_val),
        "snr": float(snr_val),
        "rango_dinamico": float(rango_dinamico)
    }

def clasificar_audio(y, umbrales):
    """Clasifica el sonido en vivo comparándolo con los valores medios normalizados."""
    analisis = analizar_audio(y)
    
    def distancia(tipo):
        """Calcula la distancia euclidiana ponderada con énfasis en entropía, curtosis y SNR."""
        return np.sqrt(sum([
            3 * (analisis["entropia_espectral"] - umbrales[tipo].get("entropia_espectral", 0)) ** 2,  # Aumentamos el peso de la entropía
            2 * (analisis["curtosis"] - umbrales[tipo].get("curtosis_media", 0)) ** 2,  # Curtosis es más importante
            2 * (analisis["snr"] - umbrales[tipo].get("snr_media", 0)) ** 2,  # SNR ahora es más relevante
            (analisis["autocorrelacion"] - umbrales[tipo].get("autocorrelacion_media", 0)) ** 2,
            (analisis["autocovarianza"] - umbrales[tipo].get("autocovarianza_media", 0)) ** 2,
            (analisis["skewness"] - umbrales[tipo].get("skewness_media", 0)) ** 2,
            (analisis["rango_dinamico"] - umbrales[tipo].get("rango_dinamico_media", 0)) ** 2
        ]))
    
    dist_cancion = distancia("cancion")
    dist_ruido = distancia("ruido_blanco")
    
    return "🎵 Canción" if dist_cancion < dist_ruido else "🔊 Ruido Blanco", analisis

def detectar_en_tiempo_real():
    """Captura audio en tiempo real y clasifica continuamente, mostrando valores en vivo."""
    umbrales = cargar_umbrales()
    if not umbrales:
        print("⚠️ No hay datos de referencia para clasificar el sonido.")
        return
    
    print("🎙️ Iniciando detección en tiempo real... (Presiona Ctrl+C para detener)")
    try:
        while True:
            y = capturar_audio()
            clasificacion, analisis = clasificar_audio(y, umbrales)
            print(f"➡️ {clasificacion}")
            print(f"📊 Entropía: {analisis['entropia_espectral']:.6f}, Curtosis: {analisis['curtosis']:.6f}, SNR: {analisis['snr']:.6f}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Detección detenida.")

if __name__ == "__main__":
    detectar_en_tiempo_real()
