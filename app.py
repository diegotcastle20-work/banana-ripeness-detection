import os
import base64
import traceback

# 1. Silenciar logs innecesarios de C++ de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

import time

@app.route('/save_sample', methods=['POST'])
def save_sample():
    """Guarda la imagen actual en la carpeta local según la clase seleccionada"""
    try:
        data = request.get_json()
        img_base64 = data.get('image_data')
        label = data.get('label')  # 'sano' o 'mal_estado'

        if not img_base64 or not label:
            return jsonify({'success': False, 'error': 'Datos incompletos.'}), 400

        # Remover prefijo Base64
        if "," in img_base64:
            header, encoded = img_base64.split(",", 1)
        else:
            encoded = img_base64

        img_bytes = base64.b64decode(encoded)

        # Crear subdirectorio si no existe
        folder_path = os.path.join("dataset_propio", label)
        os.makedirs(folder_path, exist_ok=True)

        # Crear nombre único con milisegundos
        filename = f"{label}_{int(time.time() * 1000)}.jpg"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "wb") as f:
            f.write(img_bytes)

        print(f"📁 Imagen guardada con éxito en: {file_path}")
        return jsonify({'success': True, 'message': f'Muestra guardada en {label}'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ==========================================================================
# 2. CARGA Y CONFIGURACIÓN DEL MODELO TFLITE
# ==========================================================================
MODEL_PATH = "modelo_platano_int8.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR CRÍTICO: No se encontró '{MODEL_PATH}' en la carpeta raíz.")
else:
    print(f"📦 Cargando modelo TFLite: {MODEL_PATH}")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    IMG_HEIGHT = input_shape[1]  # 96
    IMG_WIDTH = input_shape[2]   # 96
    INPUT_DTYPE = input_details[0]['dtype']
    OUTPUT_SHAPE = output_details[0]['shape']

    print(f"✅ Configuración del Modelo Cargada:")
    print(f"   - Entrada Requerida: {IMG_WIDTH}x{IMG_HEIGHT} px | Tipo: {INPUT_DTYPE}")
    print(f"   - Salida de la Red: {OUTPUT_SHAPE}")

except Exception as e:
    print(f"❌ Error al inicializar TFLite: {e}")


def preprocess_image_to_96x96(img):
    """
    Recorta el centro de la imagen para que sea cuadrada (1:1) 
    y luego la redimensiona exactamente a 96x96 px sin deformar el plátano.
    """
    h, w, _ = img.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # Recorte central cuadrado
    img_square = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    
    # Redimensionar a 96x96
    img_96 = cv2.resize(img_square, (IMG_WIDTH, IMG_HEIGHT))
    
    # Convertir espacio de color BGR (OpenCV) -> RGB (Keras)
    img_rgb = cv2.cvtColor(img_96, cv2.COLOR_BGR2RGB)
    
    return img_rgb


# ==========================================================================
# CONFIGURACIÓN DE ETIQUETAS Y CLASES
# ==========================================================================
# Si la red entrega 1.0 para SANO, mantén INVERTIR_CLASES = False
# Si entrega 1.0 para MAL ESTADO, cambia a True
INVERTIR_CLASES = False 

def model_predict_classification(img_np):
    """
    Inferencia corregida: Sin doble sigmoide y con normalización INT8 exacta
    """
    # 1. Recorte central cuadrado y redimensionado a 96x96
    img_cropped = preprocess_image_to_96x96(img_np) # Retorna RGB de 96x96

    # 2. Normalización y cuantización directa para TFLite INT8
    input_scale, input_zero_point = input_details[0]['quantization']
    
    if input_scale > 0:
        # Convertir directamente de [0, 255] a la escala INT8 del modelo
        input_data = (img_cropped.astype(np.float32) / input_scale + input_zero_point).astype(INPUT_DTYPE)
    else:
        # Si el modelo espera Float32 normalizado [-1, 1]
        img_float = (img_cropped.astype(np.float32) / 127.5) - 1.0
        input_data = img_float.astype(INPUT_DTYPE)

    input_data = np.expand_dims(input_data, axis=0)

    # 3. Inferencia TFLite
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Lectura de salida
    output_data = interpreter.get_tensor(output_details[0]['index'])
    raw_val = float(output_data[0][0])

    # 5. Decuantización del resultado
    output_scale, output_zero_point = output_details[0]['quantization']
    
    if output_scale > 0:
        prob_modelo = (raw_val - output_zero_point) * output_scale
    else:
        # Si el modelo entrega logits sin sigmoide
        prob_modelo = 1.0 / (1.0 + np.exp(-raw_val))

    # Asegurar rango dinámico entre 0.0 y 1.0
    prob_modelo = float(np.clip(prob_modelo, 0.0, 1.0))

    print(f"🔍 [DEBUG TENSOR] Raw: {raw_val} | Probabilidad Real: {prob_modelo*100:.1f}%")

    # 6. Asignación de clases
    if INVERTIR_CLASES:
        prob_bad = prob_modelo
        prob_sano = 1.0 - prob_bad
    else:
        prob_sano = prob_modelo
        prob_bad = 1.0 - prob_sano

    return {
        "sano": prob_sano,
        "mal_estado": prob_bad
    }


# ==========================================================================
# 3. RUTAS DE LA APLICACIÓN FLASK
# ==========================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ninguna imagen.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo de imagen vacío.'}), 400

        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Imagen no válida.'}), 400

        probs = model_predict_classification(img)
        prob_sano = probs["sano"]
        prob_bad = probs["mal_estado"]

        status = "sano" if prob_sano >= prob_bad else "mal_estado"

        print(f"📊 [Webcam] Predicción -> Sano: {prob_sano*100:.1f}% | Mal Estado: {prob_bad*100:.1f}%")

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
        print("❌ Error en /predict:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict_esp32', methods=['POST'])
def predict_esp32():
    try:
        data = request.get_json()
        esp32_ip = data.get('ip', '').strip()

        if not esp32_ip:
            return jsonify({'success': False, 'error': 'Ingresa la IP de la ESP32.'}), 400

        # Limpiar prefijos si el usuario escribió http:// o https://
        esp32_ip = esp32_ip.replace("http://", "").replace("https://", "").strip("/")

        target_url = f"http://{esp32_ip}/capture"
        print(f"📡 Solicitando captura a: {target_url}")

        # Configurar proxies=None para ignorar la redirección de Ngrok/Sistema
        resp = requests.get(target_url, timeout=5, proxies={"http": None, "https": None})

        if resp.status_code != 200:
            return jsonify({'success': False, 'error': 'No se pudo obtener imagen de la ESP32-CAM.'}), 400

        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Imagen corrupta recibida de la ESP32.'}), 400

        # Inferencia TFLite real
        probs = model_predict_classification(img)
        prob_sano = probs["sano"]
        prob_bad = probs["mal_estado"]

        status = "sano" if prob_sano >= prob_bad else "mal_estado"

        print(f"📊 [ESP32] Predicción -> Sano: {prob_sano*100:.1f}% | Mal Estado: {prob_bad*100:.1f}%")

        # Intentar notificar al servomotor ignorando proxies
        try:
            action_url = f"http://{esp32_ip}/action"
            requests.post(action_url, json={"status": status}, timeout=2, proxies={"http": None, "https": None})
            print(f"⚙️ Orden de servomotor ('{status}') enviada con éxito a la ESP32.")
        except Exception as ex:
            print(f"⚠️ No se pudo enviar señal de servo a la ESP32: {ex}")

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
        print("❌ Error en /predict_esp32:")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f"Error de conexión con la ESP32: {str(e)}"}), 500


if __name__ == '__main__':
    print("\n🚀 Servidor listo en http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)


# .\ngrok.exe config add-authtoken 3Jc34ZETjGcVsf09Del7Qh4Q0QB_6DnnCRfnxJsAS5dJgbSdU
# py app.py
# .\ngrok.exe http 
# .\ngrok.exe http 5000 --log=stdout 