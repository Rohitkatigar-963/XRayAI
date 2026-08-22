const API_BASE = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", () => {

  const loginForm = document.getElementById("loginForm");
  const registerForm = document.getElementById("registerForm");

  // ================= REGISTER =================
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const name = document.getElementById("regName").value;
      const email = document.getElementById("regEmail").value;
      const password = document.getElementById("regPassword").value;

      try {
        const response = await fetch(`${API_BASE}/register`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({ name, email, password })
        });

        const data = await response.json();

        if (!response.ok) {
          alert(data.detail || "Registration failed");
          return;
        }

        alert("Registration successful! Please login.");
        window.location.href = "login.html";

      } catch (err) {
        alert("Server error. Is backend running?");
      }
    });
  }

  // ================= LOGIN =================
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const email = document.getElementById("loginEmail").value;
      const password = document.getElementById("loginPassword").value;

      try {
        const formData = new URLSearchParams();
        formData.append("username", email);
        formData.append("password", password);

        const response = await fetch(`${API_BASE}/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/x-www-form-urlencoded"
          },
          body: formData
        });

        const data = await response.json();

        if (!response.ok) {
          alert(data.detail || "Login failed");
          return;
        }

        // Save JWT token
        localStorage.setItem("token", data.access_token);

        window.location.href = "index.html";

      } catch (err) {
        alert("Server error. Is backend running?");
      }
    });
  }

});