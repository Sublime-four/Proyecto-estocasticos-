import pyaudio
import wave
import json
import time
import os

# Configuración de grabación
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
DURATION = 3  # Duración de cada grabación en segundos
INTERVAL = 4  # Intervalo entre grabaciones
OUTPUT_FOLDER = "recordings"
METADATA_FILE = "recordings_metadata.json"

# Crear la carpeta de grabaciones si no existe
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def grabar_audio(nombre_archivo):
    """Graba un fragmento de audio y lo guarda en un archivo .wav."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK)
    
    print(f"🎙️ Grabando: {nombre_archivo}...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Guardar en archivo WAV
    ruta_archivo = os.path.join(OUTPUT_FOLDER, nombre_archivo)
    with wave.open(ruta_archivo, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
    
    print(f"✅ Guardado en: {ruta_archivo}")
    return ruta_archivo

def guardar_metadata(nuevos_datos):
    """Guarda los metadatos de las grabaciones en un archivo JSON sin sobrescribir datos previos."""
    try:
        with open(METADATA_FILE, "r") as f:
            datos_existentes = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        datos_existentes = []
    
    datos_existentes.extend(nuevos_datos)
    
    with open(METADATA_FILE, "w") as f:
        json.dump(datos_existentes, f, indent=4)
    print("📄 Metadatos actualizados.")

def iniciar_grabacion(tipo_senal, num_grabaciones):
    """Realiza varias grabaciones y guarda los metadatos."""
    metadata = []
    for i in range(num_grabaciones):
        timestamp = int(time.time())
        nombre_archivo = f"{tipo_senal}_{timestamp}.wav"
        ruta = grabar_audio(nombre_archivo)
        
        metadata.append({
            "tipo": tipo_senal,
            "archivo": nombre_archivo,
            "ruta": ruta,
            "timestamp": timestamp
        })
        time.sleep(INTERVAL)
    
    guardar_metadata(metadata)

if __name__ == "__main__":
    print("Seleccione el tipo de grabación:")
    print("1. Canción 🎵")
    print("2. Ruido blanco 🔊")
    opcion = input("Ingrese el número de la opción: ")
    
    tipo_senal = "cancion" if opcion == "1" else "ruido_blanco"
    num_grabaciones = int(input("¿Cuántas grabaciones desea hacer?: "))
    
    iniciar_grabacion(tipo_senal, num_grabaciones)
    print("📌 Grabaciones finalizadas y almacenadas.")