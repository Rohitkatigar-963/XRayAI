const token = localStorage.getItem("token");

if (!token) {
    window.location.href = "login.html";
}

async function loadProfile() {

    try {

        const userResponse = await fetch(
            "http://127.0.0.1:8000/me",
            {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            }
        );

        const user = await userResponse.json();

        const historyResponse = await fetch(
            "http://127.0.0.1:8000/history",
            {
                headers: {
                    Authorization: `Bearer ${token}`
                }
            }
        );

        const scans = await historyResponse.json();

        document.getElementById("profileName").innerText =
            user.name;

        document.getElementById("profileEmail").innerText =
            user.email;

            document.getElementById("accountName").innerText = user.name;

            document.getElementById("accountEmail").innerText = user.email;
            
            document.getElementById("accountId").innerText = "#" + user.id;   

        // Initials
        const parts = user.name.trim().split(" ");

        let initials = "";

        if (parts.length === 1) {
            initials = parts[0][0];
        } else {
            initials =
                parts[0][0] +
                parts[parts.length - 1][0];
        }

        document.getElementById(
            "profileCircle"
        ).innerText =
            initials.toUpperCase();

        // Statistics
        document.getElementById(
            "profileScans"
        ).innerText =
            scans.length;

        document.getElementById(
            "profileNormal"
        ).innerText =
            scans.filter(
                s => s.prediction.toLowerCase() === "normal"
            ).length;

        document.getElementById(
            "profilePneumonia"
        ).innerText =
            scans.filter(
                s => s.prediction.toLowerCase() === "pneumonia"
            ).length;

        document.getElementById(
            "profileCovid"
        ).innerText =
            scans.filter(
                s => s.prediction.toLowerCase() === "covid"
            ).length;

    }
    catch (err) {
        console.error(err);
    }
}

document
.getElementById("logoutBtn")
.addEventListener(
    "click",
    () => {

        localStorage.removeItem("token");

        window.location.href =
            "login.html";
    }
);

loadProfile();