const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultDiv = document.getElementById("result");
const previewDiv = document.getElementById("preview");

const predictionSpan = document.getElementById("prediction");
const confidenceSpan = document.getElementById("confidence");
const riskSpan = document.getElementById("risk");

const covidBar = document.getElementById("covidBar");
const pneumoniaBar = document.getElementById("pneumoniaBar");
const normalBar = document.getElementById("normalBar");

const covidPct = document.getElementById("covidPct");
const pneumoniaPct = document.getElementById("pneumoniaPct");
const normalPct = document.getElementById("normalPct");

const covidGraph = document.getElementById("covidGraph");
const pneumoniaGraph = document.getElementById("pneumoniaGraph");
const normalGraph = document.getElementById("normalGraph");

const covidGraphPct = document.getElementById("covidGraphPct");
const pneumoniaGraphPct = document.getElementById("pneumoniaGraphPct");
const normalGraphPct = document.getElementById("normalGraphPct");

// Preview image
imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    previewDiv.innerHTML = `<img src="${reader.result}" />`;
  };
  reader.readAsDataURL(file);
});

// Animate count-up %
function animateCount(el, target) {
  let current = 0;
  const value = parseFloat(target);
  const step = Math.max(value / 30, 1);

  const timer = setInterval(() => {
    current += step;
    if (current >= value) {
      current = value;
      clearInterval(timer);
    }
    el.innerText = current.toFixed(1) + "%";
  }, 20);
}

// Analyze click
analyzeBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) {
    alert("Please select an X-ray image first.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    analyzeBtn.disabled = true;
    analyzeBtn.innerText = "Analyzing...";

    const token = localStorage.getItem("token");

const res = await fetch(API_URL, {
  method: "POST",
  headers: {
    "Authorization": `Bearer ${token}`
  },
  body: formData
});

if (!res.ok) {
  let errorMessage = `Request failed (${res.status})`;

  try {
    const errorData = await res.json();
    errorMessage = errorData.detail || errorMessage;
  } catch (e) {
    // Response was not JSON
  }

  // Login/session problem
  if (res.status === 401) {
    localStorage.removeItem("token");
    alert("❌ Your login session is invalid or expired. Please login again.");
    window.location.href = "login.html";
    return;
  }

  throw new Error(errorMessage);
}

const data = await res.json();

    predictionSpan.innerText = data.prediction;
    confidenceSpan.innerText = data.confidence.toFixed(2) + "%";
    riskSpan.innerText = data.risk;

    const c = data.probabilities.COVID.toFixed(1);
    const p = data.probabilities.PNEUMONIA.toFixed(1);
    const n = data.probabilities.NORMAL.toFixed(1);

    covidBar.style.width = `${c}%`;
    pneumoniaBar.style.width = `${p}%`;
    normalBar.style.width = `${n}%`;

    covidGraph.style.height = `${c}%`;
    pneumoniaGraph.style.height = `${p}%`;
    normalGraph.style.height = `${n}%`;

    animateCount(covidPct, c);
    animateCount(pneumoniaPct, p);
    animateCount(normalPct, n);

    animateCount(covidGraphPct, c);
    animateCount(pneumoniaGraphPct, p);
    animateCount(normalGraphPct, n);

    resultDiv.classList.remove("hidden");
  } catch (err) {
    console.error("X-ray analysis error:", err);
  
    if (err instanceof TypeError) {
      alert(
        "❌ Could not connect to the backend.\n\n" +
        "Please make sure FastAPI is running on http://127.0.0.1:8000"
      );
    } else {
      alert("❌ Analysis failed:\n\n" + err.message);
    }
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerText = "Analyze X-ray";
  }
});

const downloadBtn = document.getElementById("downloadPdfBtn");

if (downloadBtn) {
    downloadBtn.addEventListener("click", async () => {

        const token = localStorage.getItem("token");

        const response = await fetch(
            "http://127.0.0.1:8000/download-report",
            {
                method: "GET",
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );

        if (!response.ok) {
            alert("Could not generate PDF report.");
            return;
        }

        const blob = await response.blob();

        const url = window.URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = "xray_report.pdf";

        document.body.appendChild(a);
        a.click();

        a.remove();
        window.URL.revokeObjectURL(url);
    });
}






const logoutBtn = document.getElementById("logoutBtn");

if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {

        localStorage.removeItem("token");

        window.location.href = "login.html";
    });
}


async function loadProfileAvatar() {

  const token = localStorage.getItem("token");

  if (!token) return;

  const response = await fetch(
      "http://127.0.0.1:8000/me",
      {
          headers: {
              Authorization: `Bearer ${token}`
          }
      }
  );

  if (!response.ok) return;

  const user = await response.json();

  const avatar =
      document.getElementById("profileAvatar");

  if (!avatar) return;

  const parts = user.name.trim().split(" ");

  let initials = "";

  if (parts.length === 1) {
      initials = parts[0][0];
  } else {
      initials =
          parts[0][0] +
          parts[parts.length - 1][0];
  }

  avatar.innerText =
      initials.toUpperCase();
}

loadProfileAvatar();




