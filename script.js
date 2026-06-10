const tabs = document.querySelectorAll(".tab");
const tables = document.querySelectorAll(".table-shell");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    tables.forEach((table) => table.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.table).classList.add("active");
  });
});

const copyButton = document.getElementById("copyBibtex");
copyButton.addEventListener("click", async () => {
  await navigator.clipboard.writeText(document.getElementById("bibtex").innerText);
  copyButton.textContent = "Copied";
  window.setTimeout(() => {
    copyButton.textContent = "Copy BibTeX";
  }, 1600);
});
