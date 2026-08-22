const token = localStorage.getItem("token");

if (!token) {
    window.location.href = "login.html";
}

let scans = [];

async function loadHistory() {

    const response = await fetch(
        "http://127.0.0.1:8000/history",
        {
            headers: {
                Authorization: `Bearer ${token}`
            }
        }
    );

    scans = await response.json();

    updateStats();

    render(scans);
}

function updateStats() {

    document.getElementById("totalScans").innerText =
        scans.length;

    document.getElementById("normalCount").innerText =
        scans.filter(
            s => s.prediction.toLowerCase() === "normal"
        ).length;

    document.getElementById("pneumoniaCount").innerText =
        scans.filter(
            s => s.prediction.toLowerCase() === "pneumonia"
        ).length;

    document.getElementById("covidCount").innerText =
        scans.filter(
            s => s.prediction.toLowerCase() === "covid"
        ).length;
}

function render(data) {

    const container =
        document.getElementById("historyContainer");

    container.innerHTML = "";

    data.forEach(scan => {

        let color = "#38bdf8";

        if (scan.prediction.toLowerCase() === "normal")
            color = "#22c55e";

        if (scan.prediction.toLowerCase() === "pneumonia")
            color = "#ef4444";

        if (scan.prediction.toLowerCase() === "covid")
            color = "#f97316";

        container.innerHTML += `

        <div class="glass history-card">

            <div class="history-top">

                <div>
                    <h2>${scan.image_name}</h2>

                    <p class="history-date">
                        ${new Date(
                            scan.timestamp
                        ).toLocaleString()}
                    </p>
                </div>

                <div
                    class="prediction-badge"
                    style="background:${color}"
                >
                    ${scan.prediction}
                </div>

            </div>

            <div class="history-confidence">

                <span>Confidence</span>

                <span>
                    ${scan.confidence.toFixed(2)}%
                </span>

            </div>

            <div class="bar-track">

                <div
                    class="bar-fill"
                    style="
                        width:${scan.confidence}%;
                        background:${color};
                    "
                ></div>

            </div>

            <div class="graph-3d">

    <div class="graph-bar covid-3d">
        <span>COVID</span>
        <div
            class="graph-fill"
            style="height:${scan.covid_probability}%"
        >
            <span class="graph-percent">
                ${scan.covid_probability.toFixed(1)}%
            </span>
        </div>
    </div>

    <div class="graph-bar pneumonia-3d">
        <span>PNEUMONIA</span>
        <div
            class="graph-fill"
            style="height:${scan.pneumonia_probability}%"
        >
            <span class="graph-percent">
                ${scan.pneumonia_probability.toFixed(1)}%
            </span>
        </div>
    </div>

    <div class="graph-bar normal-3d">
        <span>NORMAL</span>
        <div
            class="graph-fill"
            style="height:${scan.normal_probability}%"
        >
            <span class="graph-percent">
                ${scan.normal_probability.toFixed(1)}%
            </span>
        </div>
    </div>

</div>

            <div style="display:flex; gap:15px; flex-wrap:wrap;">

    <button
        class="analyze-btn"
        onclick="downloadPDF(${scan.id})"
    >
        Download PDF
    </button>

    <button
        class="delete-btn"
        onclick="deleteScan(${scan.id})"
    >
        🗑 Delete
    </button>

</div>

        </div>
        `;
    });
}


async function downloadPDF(predictionId) {

    console.log("Prediction ID =", predictionId);

    const response = await fetch(
        `http://127.0.0.1:8000/download-report/${predictionId}`,
        {
            headers: {
                Authorization: `Bearer ${token}`
            }
        }
    );

    if (!response.ok) {
        alert("Failed to download report");
        return;
    }

    const blob = await response.blob();

    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");

    a.href = url;
    a.download = `report_${predictionId}.pdf`;

    document.body.appendChild(a);

    a.click();

    a.remove();

    window.URL.revokeObjectURL(url);
}


async function deleteScan(predictionId) {

    const confirmed = confirm(
        "Are you sure you want to delete this scan?"
    );

    if (!confirmed) return;

    const response = await fetch(
        `http://127.0.0.1:8000/history/${predictionId}`,
        {
            method: "DELETE",
            headers: {
                Authorization: `Bearer ${token}`
            }
        }
    );

    if (!response.ok) {
        alert("Failed to delete scan.");
        return;
    }

    scans = scans.filter(
        scan => scan.id !== predictionId
    );

    updateStats();
    render(scans);
} 
document
.getElementById("searchBox")
.addEventListener(
    "input",
    e => {

        const query =
            e.target.value.toLowerCase();

        const filtered =
            scans.filter(
                s =>
                s.image_name
                .toLowerCase()
                .includes(query)
            );

        render(filtered);
    }
);

loadHistory();

