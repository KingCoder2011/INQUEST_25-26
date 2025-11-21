const toggle = document.getElementById("modeToggle");
toggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  // Change button icon on toggle
  toggle.textContent = document.body.classList.contains("dark") ? "☀️" : "🌙";
});
        