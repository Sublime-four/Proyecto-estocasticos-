# Proyecto: Detección de Ruido Blanco y Señales Normales

## Descripción
Este proyecto consiste en un sistema desarrollado en Python para reconocer y clasificar audio como ruido blanco o señal normal. Utiliza técnicas de procesamiento de señales, incluyendo autocovarianza, autocorrelación y análisis de espectro, para identificar patrones en los datos de audio. El sistema se entrena previamente y luego es capaz de clasificar nuevas muestras.

## Características
- Clasificación de audio en dos categorías: ruido blanco y señal normal.
- Fase separada de entrenamiento y prueba.
- Uso de autocovarianza, autocorrelación y análisis espectral.
- Almacenamiento de datos de audio para mejorar el modelo.

## Requisitos del Sistema
- **Lenguaje**: Python 3.8 o superior.
- **Dependencias**:
  - `numpy` para manipulación de datos.
  - `scipy` para procesamiento de señales.
  - `matplotlib` para visualización.
  - `librosa` para manejo de audio.

## Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/usuario/white-noise-detector.git
   cd white-noise-detector
   ```
2. Instalar las dependencias necesarias:
   ```bash
   pip install numpy scipy matplotlib librosa
   ```

## Uso
### 1. Fase de Entrenamiento
Ejecutar el script de entrenamiento para capturar y almacenar datos de audio:
```bash
python train_model.py
```

### 2. Fase de Clasificación
Clasificar nuevos datos de audio:
```bash
python classify_audio.py --input ejemplo.wav
```

## Ejemplo
```bash
> python classify_audio.py --input sample.wav
Resultado: Ruido Blanco
```

## Seguridad y Privacidad
- Los datos de audio capturados se almacenan de forma local.
- Se recomienda anonimizar los datos sensibles si se comparten.

## Contribuciones
Las contribuciones son bienvenidas. Por favor, abre un issue antes de realizar un pull request para discutir cambios mayores.



