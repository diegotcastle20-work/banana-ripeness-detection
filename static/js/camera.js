document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById("video");
    const processedImg = document.getElementById("processed-output");
    const canvas = document.getElementById("canvas");

    const btnToggle = document.getElementById("btn-toggle-cam");
    const btnSwitch = document.getElementById("btn-switch-cam");
    const btnSnap = document.getElementById("btn-snap");
    const fileInput = document.getElementById("file-input");

    const badgeStatus = document.getElementById("badge-status");
    const percentSano = document.getElementById("percent-sano");
    const percentBad = document.getElementById("percent-bad");
    const barSano = document.getElementById("bar-sano");
    const barBad = document.getElementById("bar-bad");
    const sorterArm = document.getElementById("sorter-arm");

    let currentStream = null;
    let currentFacingMode = "environment"; 

    // 1. Control de Cámara
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
                badgeStatus.textContent = "Cámara lista. Presiona 'Clasificar'";
                badgeStatus.className = "result-badge badge-neutral";
            }
        } catch (err) {
            console.error("Error al acceder a la cámara:", err);
            alert("No se pudo iniciar la cámara. Verifica los permisos HTTPS/Ngrok.");
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
                if (badgeStatus) badgeStatus.textContent = "Cámara desactivada";
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

    // 2. Captura y Procesamiento
    if (btnSnap) {
        btnSnap.addEventListener("click", () => {
            if (!currentStream || !video || video.videoWidth === 0) {
                alert("Primero debes activar la cámara.");
                return;
            }

            const ctx = canvas.getContext("2d");
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (blob) sendFrameToServer(blob);
            }, "image/jpeg", 0.9);
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) sendFrameToServer(file);
        });
    }

    // 3. Petición POST al servidor Flask
    async function sendFrameToServer(imageFileOrBlob) {
        if (badgeStatus) {
            badgeStatus.textContent = "⏳ Analizando clasificación...";
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
            if (badgeStatus) badgeStatus.textContent = "❌ Error de conexión";
        }
    }

    // 4. Actualización Visual de Porcentajes y Brazo
    function updateUIResults(diagnosis) {
        const probSano = diagnosis.prob_sano || 0;
        const probBad = diagnosis.prob_bad || 0;

        // Actualizar textos y barras
        percentSano.textContent = `${probSano}%`;
        percentBad.textContent = `${probBad}%`;

        barSano.style.width = `${probSano}%`;
        barBad.style.width = `${probBad}%`;

        if (diagnosis.status === "mal_estado") {
            badgeStatus.textContent = "⚠️ PLÁTANO EN MAL ESTADO";
            badgeStatus.className = "result-badge badge-bad";

            sorterArm.textContent = "Posición: ACTIVAR DESVÍO 🚨 (Desechando)";
            sorterArm.className = "arm-status arm-eject";
        } else {
            badgeStatus.textContent = "✅ PLÁTANO SANO";
            badgeStatus.className = "result-badge badge-sano";

            sorterArm.textContent = "Posición: NEUTRAL (Paso Libre)";
            sorterArm.className = "arm-status arm-center";
        }
    }
});