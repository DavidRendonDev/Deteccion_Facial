#!/bin/bash
# Script para probar el pipeline de detección facial con diferentes fuentes

echo "=========================================="
echo "Face Detection Pipeline - Video Test"
echo "=========================================="
echo ""

# Activar entorno virtual
source .venv/bin/activate

echo "Selecciona una opción:"
echo "1) Usar webcam (por defecto)"
echo "2) Usar un archivo de video"
echo "3) Descargar video de prueba y ejecutar"
echo ""
read -p "Opción (1-3): " option

case $option in
    1)
        echo ""
        echo "Iniciando con webcam..."
        echo "Presiona Q para salir"
        echo ""
        python run.py
        ;;
    2)
        echo ""
        read -p "Ruta completa del video: " video_path
        if [ -f "$video_path" ]; then
            echo ""
            echo "Iniciando con video: $video_path"
            echo "Presiona Q para salir"
            echo ""
            VIDEO_SOURCE="$video_path" python run.py
        else
            echo "Error: El archivo no existe: $video_path"
            exit 1
        fi
        ;;
    3)
        echo ""
        echo "Descargando video de prueba..."
        
        # Crear directorio para videos de prueba
        mkdir -p test_videos
        
        # Descargar un video corto de prueba
        if command -v wget &> /dev/null; then
            wget -O test_videos/sample.mp4 \
                "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" \
                2>&1 | grep -E "saved|failed|error" || true
        elif command -v curl &> /dev/null; then
            curl -L -o test_videos/sample.mp4 \
                "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
        else
            echo "Error: Necesitas wget o curl instalado"
            exit 1
        fi
        
        if [ -f "test_videos/sample.mp4" ]; then
            echo ""
            echo "Video descargado exitosamente!"
            echo "Iniciando detección..."
            echo "Presiona Q para salir"
            echo ""
            VIDEO_SOURCE=test_videos/sample.mp4 python run.py
        else
            echo "Error: No se pudo descargar el video"
            exit 1
        fi
        ;;
    *)
        echo "Opción inválida"
        exit 1
        ;;
esac
