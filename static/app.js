
const $ = (id) => document.getElementById(id);

let history = [];
let imageData = null;
let recorder = null;
let audioChunks = [];
let cameraStream = null;
let lastAnswer = "";
let busy = false;

// API authentication is optional.
// If API_TOKEN is empty in your backend .env, leave the field empty.
function getHeaders() {
    const headers = {};
    const token = $("token")?.value.trim();

    if (token) {
        headers.Authorization = `Bearer ${token}`;
    }

    return headers;
}

// Display messages safely without interpreting HTML.
function bubble(text, kind = "bot", sources = []) {
    const el = document.createElement("article");
    el.className = kind;

    const content = document.createElement("div");
    content.textContent = text;
    el.appendChild(content);

    if (sources.length) {
        const sourceBox = document.createElement("div");
        sourceBox.className = "sources";
        sourceBox.textContent = "📚 Sources";

        const list = document.createElement("ul");

        sources.slice(0, 8).forEach((source) => {
            const item = document.createElement("li");

            if (source.type === "pdf") {
                item.textContent =
                    `PDF — page ${source.page ?? "unknown"}` +
                    (source.score != null
                        ? ` · relevance ${source.score}`
                        : "");
            } else if (source.url) {
                const link = document.createElement("a");
                link.href = source.url;
                link.target = "_blank";
                link.rel = "noopener noreferrer";
                link.textContent = source.title || source.url;

                item.appendChild(link);
            } else {
                item.textContent = source.title || "Source";
            }

            list.appendChild(item);
        });

        sourceBox.appendChild(list);
        el.appendChild(sourceBox);
    }

    $("messages").appendChild(el);
    $("messages").scrollTop = $("messages").scrollHeight;

    return el;
}

// Check whether the FastAPI backend is running.
async function checkConnection() {
    const status = $("status");

    try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error("Health check failed");

        const data = await response.json();

        if (data.configured) {
            status.textContent = "● Connected";
            status.dataset.state = "connected";
        } else {
            status.textContent = "⚠ Check API configuration";
            status.dataset.state = "warning";
        }
    } catch {
        status.textContent = "● Server offline";
        status.dataset.state = "offline";
    }
}

checkConnection();
setInterval(checkConnection, 30000);

// IMAGE UPLOAD

function setImage(file) {
    if (!file) return;

    const allowedTypes = [
        "image/png",
        "image/jpeg",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        alert("Please choose a PNG, JPEG, or WebP image.");
        return;
    }

    if (file.size > 5 * 1024 * 1024) {
        alert("Image must be smaller than 5 MB.");
        return;
    }

    const reader = new FileReader();

    reader.onload = () => {
        imageData = reader.result;
        $("thumbnail").src = imageData;
        $("preview").hidden = false;
    };

    reader.onerror = () => {
        alert("Could not read the selected image.");
    };

    reader.readAsDataURL(file);
}

$("image").addEventListener("change", (event) => {
    setImage(event.target.files[0]);
});

$("remove").addEventListener("click", () => {
    imageData = null;
    $("preview").hidden = true;
    $("thumbnail").removeAttribute("src");
    $("image").value = "";
});

// CHAT

$("chatForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    if (busy) return;

    const message = $("message").value.trim();

    if (!message && !imageData) {
        alert("Enter a question or attach an image.");
        return;
    }

    if (!message) {
        alert("Please describe what you want to know about the image.");
        return;
    }

    busy = true;

    const previousHistory = history.slice(-8);

    bubble(message, "user");

    if (imageData) {
        const preview = document.createElement("img");
        preview.src = imageData;
        preview.alt = "Uploaded image";
        preview.style.cssText =
            "display:block;max-width:180px;max-height:180px;" +
            "object-fit:contain;margin:8px 0 0 auto;border-radius:12px;";

        const lastMessage = $("messages").lastElementChild;
        lastMessage.appendChild(preview);
    }

    $("message").value = "";
    $("send").disabled = true;

    const pending = bubble("Thinking… Checking your question and available sources.");

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                ...getHeaders(),
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message,
                history: previousHistory,
                image_data: imageData
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || `Request failed (${response.status})`
            );
        }

        pending.replaceChildren();

        const answerText = document.createElement("div");
        answerText.textContent = data.answer || "No answer was returned.";
        pending.appendChild(answerText);

        if (data.sources?.length) {
            const sourceBox = document.createElement("div");
            sourceBox.className = "sources";
            sourceBox.textContent = "📚 Sources";

            const list = document.createElement("ul");

            data.sources.slice(0, 8).forEach((source) => {
                const item = document.createElement("li");

                if (source.type === "pdf") {
                    item.textContent =
                        `PDF — page ${source.page ?? "unknown"}`;
                } else if (source.url) {
                    const link = document.createElement("a");
                    link.href = source.url;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = source.title || source.url;
                    item.appendChild(link);
                }

                list.appendChild(item);
            });

            sourceBox.appendChild(list);
            pending.appendChild(sourceBox);
        }

        lastAnswer = data.answer || "";

        history.push(
            { role: "user", content: message },
            { role: "assistant", content: lastAnswer }
        );

        history = history.slice(-10);

        // Save the conversation ID for backend persistence.
        if (data.conversation_id) {
            sessionStorage.setItem(
                "medicalConversationId",
                data.conversation_id
            );
        }

        // Clear the selected image after a successful request.
        imageData = null;
        $("preview").hidden = true;
        $("thumbnail").removeAttribute("src");
        $("image").value = "";

    } catch (error) {
        pending.textContent = `⚠️ ${error.message}`;

        const hint = document.createElement("p");
        hint.className = "sources";
        hint.textContent =
            "Check that your FastAPI server is running and that " +
            "your Gemini and Pinecone settings are configured.";
        pending.appendChild(hint);

    } finally {
        busy = false;
        $("send").disabled = false;
        $("messages").scrollTop = $("messages").scrollHeight;
        $("message").focus();
    }
});

// CAMERA

$("camera").addEventListener("click", async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment" }
        });

        $("video").srcObject = cameraStream;
        $("video").hidden = false;
        $("cameraActions").hidden = false;

    } catch (error) {
        alert("Camera unavailable: " + error.message);
    }
});

$("closeCamera").addEventListener("click", () => {
    if (cameraStream) {
        cameraStream.getTracks().forEach((track) => track.stop());
        cameraStream = null;
    }

    $("video").srcObject = null;
    $("video").hidden = true;
    $("cameraActions").hidden = true;
});

$("snap").addEventListener("click", () => {
    const video = $("video");
    const canvas = $("canvas");

    if (!video.videoWidth || !video.videoHeight) {
        alert("Camera is not ready yet.");
        return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob((blob) => {
        if (blob) {
            setImage(new File([blob], "camera.jpg", {
                type: "image/jpeg"
            }));
        }
    }, "image/jpeg", 0.85);

    $("closeCamera").click();
});

// MICROPHONE / SPEECH-TO-TEXT

$("mic").addEventListener("click", async () => {
    if (recorder && recorder.state === "recording") {
        recorder.stop();
        $("mic").textContent = "🎤 Voice";
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia ||
        typeof MediaRecorder === "undefined") {
        alert("Audio recording is not supported in this browser.");
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

        audioChunks = [];
        recorder = new MediaRecorder(stream);

        recorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        recorder.onstop = async () => {
            stream.getTracks().forEach((track) => track.stop());

            const blob = new Blob(audioChunks, {
                type: recorder.mimeType || "audio/webm"
            });

            const formData = new FormData();
            formData.append("file", blob, "voice.webm");

            $("mic").disabled = true;
            $("mic").textContent = "⏳ Transcribing…";

            try {
                const response = await fetch("/api/transcribe", {
                    method: "POST",
                    headers: getHeaders(),
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.detail || "Transcription failed."
                    );
                }

                $("message").value = data.text || "";
                $("message").focus();

            } catch (error) {
                alert("Voice transcription failed: " + error.message);
            } finally {
                $("mic").disabled = false;
                $("mic").textContent = "🎤 Voice";
            }
        };

        recorder.start();
        $("mic").textContent = "⏹ Stop recording";

    } catch (error) {
        alert("Microphone unavailable: " + error.message);
    }
});

// TEXT-TO-SPEECH

$("listen").addEventListener("click", async () => {
    if (!lastAnswer) {
        alert("Ask a question first.");
        return;
    }

    $("listen").disabled = true;
    $("listen").textContent = "🔊 Loading…";

    try {
        const response = await fetch("/api/speak", {
            method: "POST",
            headers: {
                ...getHeaders(),
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: lastAnswer })
        });

        if (!response.ok) {
            const data = await response.json().catch(() => ({}));
            throw new Error(data.detail || "Speech generation failed.");
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        audio.onended = () => URL.revokeObjectURL(audioUrl);
        audio.onerror = () => URL.revokeObjectURL(audioUrl);

        await audio.play();

    } catch (error) {
        alert(
            "Read-aloud is unavailable: " + error.message +
            "\nCheck the backend /api/speak endpoint."
        );
    } finally {
        $("listen").disabled = false;
        $("listen").textContent = "🔊 Read answer";
    }
});

// CLEAN UP CAMERA WHEN LEAVING THE PAGE.

window.addEventListener("beforeunload", () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
});