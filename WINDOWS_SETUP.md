# Guía de Instalación para Windows 11 🪟

Este proyecto es totalmente compatible con Windows 11. Sigue estos pasos para configurarlo desde cero.

## 1. Prerrequisitos

Antes de instalar las librerías de Python, necesitas algunas herramientas base:

### A. Python
1. Descarga e instala **Python 3.10, 3.11 o 3.12** desde [python.org](https://www.python.org/downloads/).
2. **IMPORTANTE:** Al instalar, asegúrate de marcar la casilla **"Add Python to PATH"**.

### B. Visual C++ Build Tools (Crucial para InsightFace)
Librerías como `insightface` y `numpy` a veces requieren compilar código C++.
1. Descarga las **Build Tools for Visual Studio** desde [este link oficial](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Ejecuta el instalador.
3. Selecciona la carga de trabajo: **"Desarrollo para el escritorio con C++"** (Desktop development with C++).
4. Dale a "Instalar" (esto puede tardar unos minutos).

---

## 2. Instalación del Proyecto

Abre tu terminal favorita (PowerShell, CMD o Windows Terminal) y navega a la carpeta donde quieras el proyecto.

### Clonar el repositorio
```powershell
git clone https://github.com/DavidRendonDev/Deteccion_Facial.git
cd Deteccion_Facial
```

### Crear Entorno Virtual
Es buena práctica aislar las dependencias:
```powershell
python -m venv .venv
```

### Activar el Entorno
*   **En PowerShell:**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
    *(Si te da error de permisos, ejecuta primero: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

*   **En CMD:**
    ```cmd
    .venv\Scripts\activate.bat
    ```

### Instalar Dependencias
```powershell
pip install -r requirements.txt
```

---

## 3. Ejecutar el Proyecto

### Opción A: Usar Webcam
```powershell
python run.py
```

### Opción B: Usar Video de Archivo
En Windows, las variables de entorno se definen distinto antes del comando.

*   **En PowerShell:**
    ```powershell
    $env:VIDEO_SOURCE="video.mp4"; python run.py
    ```

*   **En CMD:**
    ```cmd
    set VIDEO_SOURCE=video.mp4 && python run.py
    ```

---

## 4. Notas Adicionales

*   **DeepFace:** La primera vez que ejecutes el análisis de emociones, el programa descargará automáticamente los pesos del modelo (aprox 6MB) en `C:\Users\TU_USUARIO\.deepface`.
*   **Rendimiento:** Si tienes una tarjeta gráfica NVIDIA, puedes instalar `onnxruntime-gpu` para que vaya mucho más fluido:
    ```powershell
    pip install onnxruntime-gpu
    ```
    *(Asegúrate de cambiar `use_gpu=False` a `True` en `src/config.py` o en el `.env`).*
