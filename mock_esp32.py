import os
import io
import cv2
import numpy as np
from flask import Flask, send_file, request, jsonify

app = Flask(__name__)

servo_position = 90

@app.route('/capture', methods=['GET'])
def capture():
    """
    Envía una foto real si existe 'test_podrido.jpg' en la carpeta.
    Si no existe, genera una simulación con tonos marrón/negro realistas.
    """
    filename = "test_podrido.jpg"
    
    # 1. Si guardaste una foto real de plátano podrido en el proyecto, la envía directamente
    if os.path.exists(filename):
        return send_file(filename, mimetype='image/jpeg')

    # 2. Si no hay archivo real, genera un plátano con tonos marrón/oscuro (BGR)
    img = np.ones((240, 320, 3), dtype=np.uint8) * 230 # Fondo claro no tan saturado
    
    # Cuerpo del plátano en tono marrón oscuro de descomposición (BGR: 20, 40, 70)
    cv2.ellipse(img, (160, 120), (105, 32), 12, 0, 360, (20, 40, 70), -1)
    
    # Manchas de pudrición
    cv2.circle(img, (130, 115), 18, (10, 15, 30), -1)
    cv2.circle(img, (180, 125), 22, (5, 10, 20), -1)

    _, buffer = cv2.imencode('.jpg', img)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/jpeg')


@app.route('/action', methods=['POST'])
def action():
    """Recibe la orden de mover el servomotor desde app.py"""
    global servo_position
    data = request.get_json()
    status = data.get('status')

    if status == 'mal_estado':
        servo_position = 180
        print("\n🚨 [ESP32 SIMULADA] Plátano Mal Estado -> SERVO EN 180° (Desechando)")
    elif status == 'sano':
        servo_position = 90
        print("\n✅ [ESP32 SIMULADA] Plátano Sano -> SERVO EN 90° (Paso Libre)")

    return jsonify({'success': True, 'servo_position': servo_position})


if __name__ == '__main__':
    print("🤖 ESP32-CAM Simulada ejecutándose en http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001)