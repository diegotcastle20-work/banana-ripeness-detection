import os
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

# Inicializar la aplicación Flask
app = Flask(__name__)

# Configuración del modelo YOLO
MODEL_PATH = "best.pt"

# Verificar existencia del modelo antes de iniciar
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ advertencia: No se encontró '{MODEL_PATH}'. Cargando modelo genérico 'yolov8n.pt'.")
    MODEL_PATH = "yolov8n.pt"

print(f"📦 Cargando modelo YOLO desde: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


@app.route('/')
def index():
    """Ruta principal que sirve la interfaz web (index.html)"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Ruta para procesar la inferencia:
    - Recibe la imagen enviada desde cámara o archivo.
    - Realiza la detección con YOLOv8.
    - Genera la imagen anotada con recuadros (bounding boxes).
    - Devuelve JSON con el resultado formateado en Base64 y el conteo de detecciones.
    """
    try:
        # 1. Validar que la petición contenga el archivo 'image'
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ninguna imagen en la petición.'}), 400

        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'El archivo enviado está vacío.'}), 400

        # 2. Leer los bytes de la imagen y convertirlos a formato OpenCV (matriz NumPy)
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Formato de imagen no válido o corrupto.'}), 400

        # 3. Realizar la inferencia con el modelo YOLOv8
        results = model(img)

        # 4. Generar la imagen dibujada con las detecciones
        res_plotted = results[0].plot()

        # 5. Convertir la imagen procesada de vuelta a Base64 para enviarla al navegador
        _, buffer = cv2.imencode('.jpg', res_plotted)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 6. Extraer las etiquetas detectadas y sus porcentajes de confianza
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label_name = model.names[cls_id]
            confidence = float(box.conf[0])
            
            detections.append({
                'label': label_name,
                'confidence': round(confidence, 4)
            })

        # 7. Responder con el paquete JSON esperado por camera.js
        return jsonify({
            'success': True,
            'processed_image': f"data:image/jpeg;base64,{img_base64}",
            'detections': detections
        })

    except Exception as e:
        # Imprimir traza completa del error en la consola de Flask para depurar
        print("❌ Error crítico en el servidor al procesar la imagen:")
        traceback.print_exc()

        return jsonify({
            'success': False,
            'error': f"Error interno en la inferencia: {str(e)}"
        }), 500


if __name__ == '__main__':
    # host='0.0.0.0' permite recibir peticiones desde tu red local o servicios de túnel como Ngrok
    print("🚀 Servidor Flask ejecutándose en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

# .\ngrok.exe config add-authtoken 3Jc34ZETjGcVsf09Del7Qh4Q0QB_6DnnCRfnxJsAS5dJgbSdU
# py app.py
# .\ngrok.exe http 5000