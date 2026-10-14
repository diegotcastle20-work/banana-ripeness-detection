import os
import base64
import traceback
import cv2
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ==========================================================================
# CARGA Y CONFIGURACIÓN DEL MODELO TFLITE INT8
# ==========================================================================
MODEL_PATH = "modelo_platano_int8.tflite"

if not os.path.exists(MODEL_PATH):
    print(f" ERROR: No se encontró el archivo '{MODEL_PATH}' en la raíz del proyecto.")
else:
    print(f"📦 Cargando modelo TFLite desde: {MODEL_PATH}")

# Cargar el intérprete de TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def model_predict_classification(img_np):
    """
    Inferencia REAL utilizando el modelo MobileNetV2 INT8 (Clases Invertidas Corregidas)
    """
    # 1. Preprocesamiento: Redimensionar a 96x96 y convertir a RGB
    img_resized = cv2.resize(img_np, (96, 96))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 2. Normalización de entrada [-1.0, 1.0]
    img_float = (img_rgb.astype(np.float32) / 127.5) - 1.0
    
    # 3. Cuantización a INT8
    input_scale, input_zero_point = input_details[0]['quantization']
    if input_scale > 0:
        input_data = (img_float / input_scale + input_zero_point).astype(np.int8)
    else:
        input_data = img_float.astype(np.int8)
        
    input_data = np.expand_dims(input_data, axis=0)

    # 4. Inferencia
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 5. Obtener salida
    output_data = interpreter.get_tensor(output_details[0]['index'])
    raw_val = output_data[0][0]

    # 6. Decuantizar la salida
    output_scale, output_zero_point = output_details[0]['quantization']
    if output_scale > 0:
        raw_prob = (float(raw_val) - output_zero_point) * output_scale
    else:
        raw_prob = float(raw_val)

    # CORRECCIÓN DE CLASE:
    # La salida representa la probabilidad de MAL ESTADO (Clase 1)
    prob_sano = float(np.clip(raw_prob, 0.0, 1.0))
    prob_bad = 1.0 - prob_bad

    return {
        "sano": prob_sano,
        "mal_estado": prob_bad
    }

# ==========================================================================
# RUTAS DE LA APLICACIÓN FLASK
# ==========================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para capturas de Webcam local o archivos de la galería"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ninguna imagen.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'El archivo enviado está vacío.'}), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Formato de imagen no válido.'}), 400

        # Inferencia real con TFLite
        probs = model_predict_classification(img)
        prob_sano = probs.get("sano", 0.0)
        prob_bad = probs.get("mal_estado", 0.0)

        status = "sano" if prob_sano >= prob_bad else "mal_estado"

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'diagnosis': {
                'status': status,
                'prob_sano': round(prob_sano * 100, 1),
                'prob_bad': round(prob_bad * 100, 1)
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict_esp32', methods=['POST'])
def predict_esp32():
    """Ruta para solicitar captura directa a la ESP32-CAM en la red local"""
    try:
        data = request.get_json()
        esp32_ip = data.get('ip')  # Ejemplo: "192.168.1.50"

        if not esp32_ip:
            return jsonify({'success': False, 'error': 'No se proporcionó la IP de la ESP32-CAM.'}), 400

        # Flask solicita la foto directamente a la ESP32
        resp = requests.get(f"http://{esp32_ip}/capture", timeout=5)

        if resp.status_code != 200:
            return jsonify({'success': False, 'error': 'No se pudo obtener la imagen desde la ESP32-CAM.'}), 400

        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'La imagen obtenida de la ESP32 está corrupta.'}), 400

        # Inferencia real con TFLite
        probs = model_predict_classification(img)
        prob_sano = probs.get("sano", 0.0)
        prob_bad = probs.get("mal_estado", 0.0)

        status = "sano" if prob_sano >= prob_bad else "mal_estado"

        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'image_data': f"data:image/jpeg;base64,{img_base64}",
            'diagnosis': {
                'status': status,
                'prob_sano': round(prob_sano * 100, 1),
                'prob_bad': round(prob_bad * 100, 1)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': f"Error al conectar con la ESP32: {str(e)}"}), 500


if __name__ == '__main__':
    print("🚀 Servidor de Clasificación ejecutándose en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

    
# .\ngrok.exe config add-authtoken 3Jc34ZETjGcVsf09Del7Qh4Q0QB_6DnnCRfnxJsAS5dJgbSdU
# py app.py
# .\ngrok.exe http 
# .\ngrok.exe http 5000 --log=stdout 
