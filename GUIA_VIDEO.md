# Guía: Cómo Probar con un Video

## Método Rápido (Script Interactivo)

He creado un script que te facilita todo:

```bash
./test_with_video.sh
```

El script te dará 3 opciones:
1. Usar webcam
2. Usar tu propio video
3. Descargar un video de prueba automáticamente

## Método Manual

### 1. Con Webcam (Más Fácil)

```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar
python run.py
```

### 2. Con Tu Propio Video

```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar con tu video
VIDEO_SOURCE=/ruta/a/tu/video.mp4 python run.py
```

**Ejemplos de rutas:**
- `/home/ders/Videos/mi_video.mp4`
- `/home/ders/Descargas/video_prueba.mp4`
- `./mi_video.mp4` (si está en la carpeta del proyecto)

### 3. Descargar Video de Prueba

```bash
# Activar entorno virtual
source .venv/bin/activate

# Crear carpeta para videos
mkdir -p test_videos

# Descargar video de ejemplo (Big Buck Bunny)
wget -O test_videos/sample.mp4 \
  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

# Ejecutar con el video descargado
VIDEO_SOURCE=test_videos/sample.mp4 python run.py
```

## ¿Qué Verás?

Cuando el programa esté corriendo, la ventana mostrará:

### Elementos Visuales

1. **🟢 Cajas Verdes**: Alrededor de cada cara detectada
2. **🟡 Puntos Amarillos**: Keypoints faciales (ojos, nariz, boca)
3. **Etiquetas de Texto**: Información sobre cada cara

### Información en las Etiquetas

```
0.95 | T1 | P1
 │     │    │
 │     │    └─ Person ID (ID persistente de la persona)
 │     └────── Track ID (ID de seguimiento temporal)
 └──────────── Confianza de detección (0.0 - 1.0)
```

### Contador Superior

```
FPS: 25.3 | faces: 2
  │           │
  │           └─ Número de caras detectadas
  └───────────── Frames por segundo
```

## Controles

- **Q**: Salir del programa
- **ESC**: También cierra (en algunos sistemas)

## Formatos de Video Soportados

El programa soporta los formatos que OpenCV puede leer:
- ✅ MP4 (`.mp4`)
- ✅ AVI (`.avi`)
- ✅ MOV (`.mov`)
- ✅ MKV (`.mkv`)
- ✅ WebM (`.webm`)

## Configuración Avanzada

Puedes ajustar parámetros creando un archivo `.env`:

```bash
# Copiar ejemplo
cp .env.example .env

# Editar configuración
nano .env
```

### Parámetros Útiles

```bash
# Umbral de confianza (0.0 - 1.0)
# Valores más altos = menos detecciones pero más precisas
DET_THRESH=0.7

# Tamaño de detección (más pequeño = más rápido)
DET_W=320
DET_H=320

# Habilitar/deshabilitar características
ENABLE_TRACKING=true
ENABLE_REID=true

# GPU (si tienes CUDA instalado)
USE_GPU=false
```

## Solución de Problemas

### El video no se abre
```bash
# Verifica que el archivo existe
ls -lh /ruta/a/tu/video.mp4

# Verifica el formato
file /ruta/a/tu/video.mp4
```

### El programa va muy lento
```bash
# Reduce el tamaño de detección
DET_W=320 DET_H=320 python run.py

# O deshabilita re-identificación
ENABLE_REID=false python run.py
```

### No detecta caras
```bash
# Reduce el umbral de confianza
DET_THRESH=0.4 python run.py
```

## Ejemplos Prácticos

### Ejemplo 1: Video con Configuración Rápida
```bash
source .venv/bin/activate
DET_W=320 DET_H=320 VIDEO_SOURCE=mi_video.mp4 python run.py
```

### Ejemplo 2: Video con Alta Precisión
```bash
source .venv/bin/activate
DET_THRESH=0.8 DET_W=640 DET_H=640 VIDEO_SOURCE=mi_video.mp4 python run.py
```

### Ejemplo 3: Solo Detección (Sin Tracking ni ReID)
```bash
source .venv/bin/activate
ENABLE_TRACKING=false ENABLE_REID=false VIDEO_SOURCE=mi_video.mp4 python run.py
```

## Próximos Pasos

Una vez que veas el sistema funcionando:

1. **Prueba con diferentes videos** para ver cómo se comporta
2. **Ajusta los parámetros** en `.env` para optimizar
3. **Usa la API** para integrar en otras aplicaciones
4. **Modifica el código** para añadir nuevas funcionalidades

## ¿Necesitas Ayuda?

- Lee el [README.md](README.md) completo
- Revisa el [walkthrough.md](file:///home/ders/.gemini/antigravity/brain/da6e1264-d610-4abb-a4c9-f9e8dcf72871/walkthrough.md)
- Ejecuta `python demo.py` para verificar que todo funciona
