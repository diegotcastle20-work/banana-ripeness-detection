document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("video") || document.getElementById("webcam");
    const processedImg = document.getElementById("processed-output");
    const canvas = document.getElementById("canvas");

    const btnToggle = document.getElementById("btn-toggle-cam");
    const btnSwitch = document.getElementById("btn-switch-cam");
    const btnSnap = document.getElementById("btn-snap");
    const fileInput = document.getElementById("file-input");

    const badgeStatus = document.getElementById("badge-status");
    const countSano = document.getElementById("count-sano");
    const countBad = document.getElementById("count-bad");
    const sorterArm = document.getElementById("sorter-arm");

    let currentStream = null;
    let currentFacingMode = "user"; 

    // --- 1. CONTROL DE CÁMARA ---
    async function startCamera() {
        stopCamera();

        if (processedImg) processedImg.classList.add("hidden");
        if (video) video.classList.remove("hidden");

        let stream = null;
        const constraintsToTry = [
            { video: { facingMode: currentFacingMode } },
            { video: { facingMode: "user" } },
            { video: true }
        ];

        for (const constraints of constraintsToTry) {
            try {
                stream = await navigator.mediaDevices.getUserMedia(constraints);
                if (stream) break;
            } catch (err) {
                console.warn("Intento de cámara fallido:", err);
            }
        }

        if (!stream) {
            alert("No se pudo acceder a ninguna cámara.");
            return;
        }

        currentStream = stream;
        if (video) {
            video.srcObject = stream;
            try {
                await video.play();
                if (badgeStatus) {
                    badgeStatus.textContent = "Cámara lista. Presiona 'Capturar / Clasificar'";
                    badgeStatus.className = "result-badge badge-neutral";
                }
            } catch (e) {
                console.error("Error al reproducir video:", e);
            }
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
                if (badgeStatus) badgeStatus.textContent = "Cámara apagada";
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

    window.addEventListener("beforeunload", stopCamera);

    // --- 2. CAPTURA Y CLASIFICACIÓN DESDE CÁMARA ---
    if (btnSnap) {
        btnSnap.addEventListener("click", () => {
            if (!currentStream || !video || video.videoWidth === 0) {
                alert("Primero debes iniciar la cámara y esperar a que muestre imagen.");
                return;
            }

            // Preparar el canvas con la resolución real del video
            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convertir fotograma a blob e inferir
            canvas.toBlob((blob) => {
                if (blob) {
                    sendFrameToServer(blob);
                } else {
                    alert("Error al capturar el fotograma.");
                }
            }, "image/jpeg", 0.9);
        });
    }

    // --- 3. CLASIFICACIÓN DESDE ARCHIVO / GALERÍA ---
    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) {
                sendFrameToServer(file);
            }
        });
    }

    // --- 4. ENVÍO DE DATOS A FLASK (/predict) ---
    async function sendFrameToServer(imageFileOrBlob) {
        if (badgeStatus) {
            badgeStatus.textContent = "⏳ Analizando con YOLOv8...";
            badgeStatus.className = "result-badge badge-neutral";
        }

        const formData = new FormData();
        formData.append("image", imageFileOrBlob, "input.jpg");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error en el servidor: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                // Formatear Base64 si es necesario
                let imageSrc = data.processed_image;
                if (!imageSrc.startsWith("data:image")) {
                    imageSrc = "data:image/jpeg;base64," + imageSrc;
                }

                // Ocultar video y mostrar resultado
                if (processedImg) {
                    processedImg.src = imageSrc;
                    processedImg.classList.remove("hidden");
                }
                if (video) video.classList.add("hidden");

                updateUIResults(data.detections || []);
            } else {
                if (badgeStatus) badgeStatus.textContent = "❌ Error: " + (data.error || "Fallo en inferencia");
            }
        } catch (err) {
            console.error(err);
            if (badgeStatus) badgeStatus.textContent = "❌ Error de conexión con el servidor Python";
        }
    }

    // --- 5. ACTUALIZACIÓN DE LA INTERFAZ ---
    function updateUIResults(detections) {
        let sanoCount = 0;
        let badCount = 0;

        detections.forEach(det => {
            const label = String(det.label || det.name || det.class || "").toLowerCase();

            if (["overripe", "rotten", "en_mal_estado", "bad", "podrido", "decay"].some(term => label.includes(term))) {
                badCount++;
            } else if (["ripe", "fresh", "sano", "good", "banana", "platano"].some(term => label.includes(term))) {
                sanoCount++;
            }
        });

        if (countSano) countSano.textContent = sanoCount;
        if (countBad) countBad.textContent = badCount;

        if (badCount > 0) {
            if (badgeStatus) {
                badgeStatus.textContent = "⚠️ ¡PLÁTANO EN MAL ESTADO DETECTADO!";
                badgeStatus.className = "result-badge badge-bad";
            }
            if (sorterArm) {
                sorterArm.textContent = "Posición: DESVIAR SEPARADOR 🚨 (Desechar)";
                sorterArm.className = "arm-status arm-eject";
            }
        } else if (sanoCount > 0) {
            if (badgeStatus) {
                badgeStatus.textContent = "✅ PLÁTANO SANO DETECTADO";
                badgeStatus.className = "result-badge badge-sano";
            }
            if (sorterArm) {
                sorterArm.textContent = "Posición: NEUTRAL (Continuar)";
                sorterArm.className = "arm-status arm-center";
            }
        } else {
            if (badgeStatus) {
                badgeStatus.textContent = "No se detectaron plátanos claros";
                badgeStatus.className = "result-badge badge-neutral";
            }
            if (sorterArm) {
                sorterArm.textContent = "Posición: NEUTRAL (Continuar)";
                sorterArm.className = "arm-status arm-center";
            }
        }
    }
});