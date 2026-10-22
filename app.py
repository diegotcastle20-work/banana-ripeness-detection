import os
import base64
import traceback
import time
import cv2
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

# --------------------------------------------------------------------------
# 1. CONFIGURACIÓN DEL ENTORNO Y SILENCIADO DE LOGS
# --------------------------------------------------------------------------
# Desactiva avisos internos de C++ de TensorFlow para mantener la consola limpia
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# --------------------------------------------------------------------------
# 2. CONFIGURACIÓN DEL MODELO DE IA Y UMBRALES DE CONTROL
# --------------------------------------------------------------------------
MODEL_PATH = "modelo_platano_int8.tflite"

# UMBRAL DE DECISIÓN: Exige un 70% (0.70) de certeza mínima para clasificar como 'sano'.
# Si el modelo duda (ej. 60% sano), automáticamente lo desvía como 'mal_estado' por seguridad de calidad.
UMBRAL_SANO = 0.70 

# Si la salida 1.0 del modelo representa 'sano', mantén False.
INVERTIR_CLASES = False 

# Carga e inicialización del intérprete TFLite para inferencia en CPU
if not os.path.exists(MODEL_PATH):
    print(f"❌ ERROR CRÍTICO: No se encontró '{MODEL_PATH}' en la carpeta raíz.")
else:
    print(f"📦 Cargando modelo TFLite: {MODEL_PATH}")

try:
    # Carga el archivo .tflite en memoria y asigna los tensores
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Obtiene las especificaciones técnicas de los tensores de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Extrae las dimensiones requeridas por la red MobileNetV2 (96x96 px)
    input_shape = input_details[0]['shape']
    IMG_HEIGHT = input_shape[1]  # 96
    IMG_WIDTH = input_shape[2]   # 96
    INPUT_DTYPE = input_details[0]['dtype']
    OUTPUT_SHAPE = output_details[0]['shape']

    print(f"✅ Configuración del Modelo Cargada:")
    print(f"   - Entrada: {IMG_WIDTH}x{IMG_HEIGHT} px | Tipo: {INPUT_DTYPE}")
    print(f"   - Salida: {OUTPUT_SHAPE}")

except Exception as e:
    print(f"❌ Error al inicializar TFLite: {e}")


def preprocess_image_to_96x96(img):
    """
    PREPROCESAMIENTO GEOMÉTRICO (CENTER CROP 1:1):
    Recorta el centro de la imagen para convertirla en un cuadrado perfecto antes 
    de redimensionarla a 96x96 px. Evita que la fruta se aplane o deforme.
    """
    h, w, _ = img.shape
    min_dim = min(h, w)
    
    # Puntos de inicio para el recorte central
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    
    # Recorte cuadrado central en matriz Numpy
    img_square = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
    
    # Escalado a la resolución de la red (96x96)
    img_96 = cv2.resize(img_square, (IMG_WIDTH, IMG_HEIGHT))
    
    # Conversión de espacio de color: OpenCV lee en BGR, Keras/TFLite procesa en RGB
    img_rgb = cv2.cvtColor(img_96, cv2.COLOR_BGR2RGB)
    
    return img_rgb


def model_predict_classification(img_np):
    """
    INFERENCIA INT8 Y DECUANTIZACIÓN DIRECTA[cite: 1]:
    Aplica la matriz de escala y punto cero del modelo cuantizado para predecir.
    """
    # 1. Preprocesamiento geométrico
    img_cropped = preprocess_image_to_96x96(img_np)

    # 2. Mapeo del rango de píxeles [0, 255] a los parámetros de cuantización INT8[cite: 1]
    input_scale, input_zero_point = input_details[0]['quantization']
    
    if input_scale > 0:
        # Conversión directa INT8: (Pixel / Scale) + ZeroPoint[cite: 1]
        input_data = (img_cropped.astype(np.float32) / input_scale + input_zero_point).astype(INPUT_DTYPE)
    else:
        # Modo de respaldo si el modelo fuera Float32 [-1.0, 1.0][cite: 1]
        img_float = (img_cropped.astype(np.float32) / 127.5) - 1.0
        input_data = img_float.astype(INPUT_DTYPE)

    # Añadir dimensión de lote (Batch Dimension): de (96, 96, 3) a (1, 96, 96, 3)
    input_data = np.expand_dims(input_data, axis=0)

    # 3. Ejecutar la inferencia en TFLite[cite: 1]
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # 4. Obtener la salida cruda del tensor[cite: 1]
    output_data = interpreter.get_tensor(output_details[0]['index'])
    raw_val = float(output_data[0][0])

    # 5. Decuantización matemática de la salida a probabilidad real[cite: 1]
    output_scale, output_zero_point = output_details[0]['quantization']
    
    if output_scale > 0:
        # Fórmula inversa de cuantización: Prob = (Raw - ZeroPoint) * Scale[cite: 1]
        prob_modelo = (raw_val - output_zero_point) * output_scale
    else:
        # Función sigmoide para logits flotantes[cite: 1]
        prob_modelo = 1.0 / (1.0 + np.exp(-raw_val))

    # Asegura que el valor esté acotado numéricamente entre 0.0 y 1.0[cite: 1]
    prob_modelo = float(np.clip(prob_modelo, 0.0, 1.0))

    # 6. Asignación de probabilidades de clase
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

# --------------------------------------------------------------------------
# 3. RUTAS Y ENDPOINTS DEL SERVIDOR FLASK
# --------------------------------------------------------------------------

@app.route('/')
def index():
    """Sirve la página web principal del cliente[cite: 1]."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Ruta para clasificar imágenes enviadas desde la Webcam local o Galería[cite: 1]."""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No se recibió ninguna imagen.'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Archivo vacío.'}), 400

        # Decodificación del stream binario a matriz de OpenCV
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Imagen no válida.'}), 400

        # Inferencia con TFLite
        probs = model_predict_classification(img)
        prob_sano = probs["sano"]
        prob_bad = probs["mal_estado"]

        # EVALUACIÓN CON UMBRAL DE CALIDAD AJUSTADO
        status = "sano" if prob_sano >= UMBRAL_SANO else "mal_estado"

        # Re-codificar imagen para enviarla de regreso a la interfaz en formato Base64[cite: 1]
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
    """Orquestador IoT: Solicita frame a la ESP32-CAM, clasifica y envía orden al servomotor[cite: 1]."""
    try:
        data = request.get_json()
        esp32_ip = data.get('ip', '').strip().replace("http://", "").replace("https://", "").strip("/")

        if not esp32_ip:
            return jsonify({'success': False, 'error': 'IP de la ESP32 no ingresada.'}), 400

        # 1. Petición HTTP a la ESP32-CAM para capturar un frame JPEG[cite: 1]
        target_url = f"http://{esp32_ip}/capture"
        
        # proxies={"http": None} es vital para evitar que el tráfico a la red local se bloquee en Ngrok[cite: 1]
        resp = requests.get(target_url, timeout=5, proxies={"http": None, "https": None})

        if resp.status_code != 200:
            return jsonify({'success': False, 'error': 'No se obtuvo respuesta de la ESP32-CAM.'}), 400

        # Convertir bytes a imagen de OpenCV
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Imagen corrupta recibida de la ESP32.'}), 400

        # 2. Clasificación con el modelo TFLite
        probs = model_predict_classification(img)
        prob_sano = probs["sano"]
        prob_bad = probs["mal_estado"]

        # EVALUACIÓN CON UMBRAL DE CALIDAD AJUSTADO
        status = "sano" if prob_sano >= UMBRAL_SANO else "mal_estado"

        # 3. Notificación HTTP al Firmware para mover el servomotor según la clasificación[cite: 1]
        try:
            action_url = f"http://{esp32_ip}/action"
            requests.post(action_url, json={"status": status}, timeout=2, proxies={"http": None, "https": None})
            print(f"⚙️ Comando '{status}' enviado exitosamente al servomotor.")
        except Exception as ex:
            print(f"⚠️ Alerta: No se pudo enviar señal a la ESP32: {ex}")

        # Retornar respuesta al navegador cliente
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
        return jsonify({'success': False, 'error': f"Error de conexión: {str(e)}"}), 500


@app.route('/save_sample', methods=['POST'])
def save_sample():
    """Guarda imágenes capturadas en carpetas locales para reentrenar el modelo[cite: 1]."""
    try:
        data = request.get_json()
        img_base64 = data.get('image_data')
        label = data.get('label')  # 'sano' o 'mal_estado'

        if not img_base64 or not label:
            return jsonify({'success': False, 'error': 'Datos incompletos.'}), 400

        if "," in img_base64:
            _, encoded = img_base64.split(",", 1)
        else:
            encoded = img_base64

        img_bytes = base64.b64decode(encoded)

        # Creación automática de directorios del dataset[cite: 1]
        folder_path = os.path.join("dataset_propio", label)
        os.makedirs(folder_path, exist_ok=True)

        filename = f"{label}_{int(time.time() * 1000)}.jpg"
        file_path = os.path.join(folder_path, filename)

        with open(file_path, "wb") as f:
            f.write(img_bytes)

        return jsonify({'success': True, 'message': f'Muestra guardada en {label}'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("\n🚀 Servidor backend listo en http://127.0.0.1:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)


# .\ngrok.exe config add-authtoken 3Jc34ZETjGcVsf09Del7Qh4Q0QB_6DnnCRfnxJsAS5dJgbSdU
# py app.py
# .\ngrok.exe http 
# .\ngrok.exe http 5000 --log=stdout