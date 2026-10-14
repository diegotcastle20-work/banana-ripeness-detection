import os
import base64
import traceback
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# NOTA: Si usas TensorFlow / PyTorch / TFLite para MobileNetV2, cargas tu modelo aquí.
# Ejemplo simulado de inferencia para clasificación (Sano vs Mal Estado)
def model_predict_classification(img_np):
    """
    Sustituye esta función con tu modelo real MobileNetV2 (PyTorch, TF o TFLite Micro).
    Recibe la matriz OpenCV y devuelve probabilidades [prob_sano, prob_bad].
    """
    # Preprocesamiento típico: resize 96x96 / 224x224 y normalización
    img_resized = cv2.resize(img_np, (96, 96))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # --- AQUÍ EJECUTAS TU INFERENCIA MOBILENETV2 ---
    # Ejemplo: prediction = model.predict(img_rgb)
    # Por ahora simulamos una clasificación basada en intensidad de verde/marrón o aleatoria limpia
    
    # Supongamos resultado de inferencia:
    # Retornamos dict con las clases y porcentajes (0.0 a 1.0)
    return {
        "sano": 0.945,      # 94.5% sano
        "mal_estado": 0.055 # 5.5% mal estado
    }

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
            return jsonify({'success': False, 'error': 'El archivo enviado está vacío.'}), 400

        # Leer imagen
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Formato de imagen no válido.'}), 400

        # Convertir imagen original a Base64 para mostrarla en el cliente
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Realizar inferencia de Clasificación
        probs = model_predict_classification(img)
        
        # Determinar diagnóstico dominante
        prob_sano = probs.get("sano", 0.0)
        prob_bad = probs.get("mal_estado", 0.0)
        
        status = "sano" if prob_sano >= prob_bad else "mal_estado"

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

if __name__ == '__main__':
    print("🚀 Servidor de Clasificación ejecutándose en http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

# .\ngrok.exe config add-authtoken 3Jc34ZETjGcVsf09Del7Qh4Q0QB_6DnnCRfnxJsAS5dJgbSdU
# py app.py
# .\ngrok.exe http 
# .\ngrok.exe http 5000 --log=stdout