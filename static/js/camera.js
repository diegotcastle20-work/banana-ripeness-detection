document.addEventListener("DOMContentLoaded", () => {
    // --- ELEMENTOS DEL DOM ---
    const video = document.getElementById("video");
    const processedImg = document.getElementById("processed-output");
    const canvas = document.getElementById("canvas");

    const btnToggle = document.getElementById("btn-toggle-cam");
    const btnSwitch = document.getElementById("btn-switch-cam");
    const btnSnap = document.getElementById("btn-snap");
    const btnSnapEsp32 = document.getElementById("btn-snap-esp32");
    const inputIp = document.getElementById("esp32-ip");
    const fileInput = document.getElementById("file-input");

    const badgeStatus = document.getElementById("badge-status");
    const percentSano = document.getElementById("percent-sano");
    const percentBad = document.getElementById("percent-bad");
    const barSano = document.getElementById("bar-sano");
    const barBad = document.getElementById("bar-bad");
    const sorterArm = document.getElementById("sorter-arm");

    let currentStream = null;
    let currentFacingMode = "environment"; 

    // =========================================================================
    // 1. GESTIÓN DE LA WEBCAM LOCAL (CELULAR / LAPTOP)
    // =========================================================================
    async function startCamera() {
        stopCamera();
        if (processedImg) processedImg.classList.add("hidden");
        if (video) video.classList.remove("hidden");

        try {
            currentStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode }
            });
            video.srcObject = currentStream;
            await video.play();
            if (badgeStatus) {
                badgeStatus.textContent = "Cámara lista. Presiona 'Clasificar Webcam'";
                badgeStatus.className = "result-badge badge-neutral";
            }
        } catch (err) {
            console.error("Error de cámara:", err);
            alert("No se pudo iniciar la webcam. Asegúrate de usar HTTPS / Ngrok.");
        }
    }

    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        if (video) video.srcObject = null;
    }

    if (btnToggle) {
        btnToggle.addEventListener("click", () => {
            if (currentStream && currentStream.active) {
                stopCamera();
                if (badgeStatus) badgeStatus.textContent = "Webcam desactivada";
            } else {
                startCamera();
            }
        });
    }

    if (btnSwitch) {
        btnSwitch.addEventListener("click", () => {
            currentFacingMode = (currentFacingMode === "environment") ? "user" : "environment";
            startCamera();
        });
    }

    // =========================================================================
    // 2. CAPTURA Y CLASIFICACIÓN DESDE WEBCAM
    // =========================================================================
    if (btnSnap) {
        btnSnap.addEventListener("click", () => {
            if (!currentStream || !video || video.videoWidth === 0) {
                alert("Primero debes iniciar la webcam.");
                return;
            }

            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (blob) sendBlobToServer(blob);
            }, "image/jpeg", 0.9);
        });
    }

    // =========================================================================
    // 3. CAPTURA Y CLASIFICACIÓN DESDE ARCHIVO LOCAL DE GALERÍA
    // =========================================================================
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) sendBlobToServer(file);
        });
    }

    // Enviar foto (Blob/Archivo) a la ruta /predict de Flask
    async function sendBlobToServer(imageFileOrBlob) {
        if (badgeStatus) {
            badgeStatus.textContent = "⏳ Analizando imagen con el modelo...";
            badgeStatus.className = "result-badge badge-neutral";
        }

        const formData = new FormData();
        formData.append("image", imageFileOrBlob, "input.jpg");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

            const data = await response.json();

            if (data.success) {
                if (processedImg) {
                    processedImg.src = data.image_data;
                    processedImg.classList.remove("hidden");
                }
                if (video) video.classList.add("hidden");

                updateUIResults(data.diagnosis);
            } else {
                if (badgeStatus) badgeStatus.textContent = "❌ Error: " + data.error;
            }
        } catch (err) {
            console.error(err);
            if (badgeStatus) badgeStatus.textContent = "❌ Error de conexión con el servidor";
        }
    }

    // =========================================================================
    // 4. CAPTURA Y CLASIFICACIÓN DESDE ESP32-CAM POR IP
    // =========================================================================
    if (btnSnapEsp32) {
        btnSnapEsp32.addEventListener("click", async () => {
            const ip = inputIp ? inputIp.value.trim() : "";
            if (!ip) {
                alert("Por favor ingresa la IP local de tu ESP32-CAM (ej: 192.168.1.50).");
                return;
            }

            if (badgeStatus) {
                badgeStatus.textContent = `⏳ Conectando con ESP32-CAM (${ip})...`;
                badgeStatus.className = "result-badge badge-neutral";
            }

            try {
                const response = await fetch("/predict_esp32", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ ip: ip })
                });

                const data = await response.json();

                if (data.success) {
                    if (processedImg) {
                        processedImg.src = data.image_data;
                        processedImg.classList.remove("hidden");
                    }
                    if (video) video.classList.add("hidden");

                    updateUIResults(data.diagnosis);
                } else {
                    if (badgeStatus) badgeStatus.textContent = "❌ Error ESP32: " + data.error;
                }
            } catch (err) {
                console.error(err);
                if (badgeStatus) badgeStatus.textContent = "❌ Error de conexión con Flask / ESP32";
            }
        });
    }

    // =========================================================================
    // 5. ACTUALIZACIÓN VISUAL DE PORCENTAJES, BARRAS Y SEPARADOR
    // =========================================================================
    function updateUIResults(diagnosis) {
        const probSano = diagnosis.prob_sano || 0;
        const probBad = diagnosis.prob_bad || 0;

        // Actualizar valores numéricos
        if (percentSano) percentSano.textContent = `${probSano}%`;
        if (percentBad) percentBad.textContent = `${probBad}%`;

        // Animar ancho de las barras
        if (barSano) barSano.style.width = `${probSano}%`;
        if (barBad) barBad.style.width = `${probBad}%`;

        // Diagnóstico dominante
        if (diagnosis.status === "mal_estado") {
            if (badgeStatus) {
                badgeStatus.textContent = `⚠️ PLÁTANO EN MAL ESTADO (${probBad}% Confianza)`;
                badgeStatus.className = "result-badge badge-bad";
            }
            if (sorterArm) {
                sorterArm.textContent = "Posición: ACTIVAR DESVÍO 🚨 (Desechando)";
                sorterArm.className = "arm-status arm-eject";
            }
        } else {
            if (badgeStatus) {
                badgeStatus.textContent = `✅ PLÁTANO SANO (${probSano}% Confianza)`;
                badgeStatus.className = "result-badge badge-sano";
            }
            if (sorterArm) {
                sorterArm.textContent = "Posición: NEUTRAL (Paso Libre)";
                sorterArm.className = "arm-status arm-center";
            }
        }
    }
});