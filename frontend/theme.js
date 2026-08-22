// Apply saved theme immediately
const savedTheme = localStorage.getItem("theme");

if (savedTheme === "light") {
    document.body.classList.add("light");
}

const themeToggle = document.getElementById("themeToggle");

if (themeToggle) {

    themeToggle.innerText =
        document.body.classList.contains("light")
        ? "☀️"
        : "🌙";

    themeToggle.addEventListener("click", () => {

        document.body.classList.toggle("light");

        const isLight =
            document.body.classList.contains("light");

        localStorage.setItem(
            "theme",
            isLight ? "light" : "dark"
        );

        themeToggle.innerText =
            isLight ? "☀️" : "🌙";
    });
}