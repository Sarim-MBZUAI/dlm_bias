(function () {
  "use strict";

  const root = document.documentElement;
  const themeToggle = document.getElementById("theme-toggle");
  const header = document.querySelector(".site-header");
  const toast = document.getElementById("toast");

  let savedTheme = null;
  try {
    savedTheme = localStorage.getItem("nobi-theme");
  } catch (error) {
    savedTheme = null;
  }
  if (savedTheme === "light" || savedTheme === "dark") {
    root.dataset.theme = savedTheme;
  }

  function activeTheme() {
    return root.dataset.theme || "dark";
  }

  function updateThemeLabel() {
    const next = activeTheme() === "dark" ? "light" : "dark";
    themeToggle.setAttribute("aria-label", `Switch to ${next} theme`);
  }

  updateThemeLabel();

  themeToggle.addEventListener("click", function () {
    const next = activeTheme() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("nobi-theme", next);
    } catch (error) {
      // The selected theme still applies for this page view.
    }
    updateThemeLabel();
  });

  function updateHeader() {
    header.classList.toggle("is-scrolled", window.scrollY > 18);
  }

  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  const revealElements = document.querySelectorAll("[data-reveal]");
  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver(function (entries, observer) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px" });

    revealElements.forEach(function (element) {
      revealObserver.observe(element);
    });
  } else {
    revealElements.forEach(function (element) {
      element.classList.add("is-visible");
    });
  }

  const tabs = Array.from(document.querySelectorAll("[data-result-tab]"));
  const panels = Array.from(document.querySelectorAll("[data-result-panel]"));

  function activateTab(tab) {
    const target = tab.dataset.resultTab;
    tabs.forEach(function (item) {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
    });
    panels.forEach(function (panel) {
      const selected = panel.dataset.resultPanel === target;
      panel.hidden = !selected;
      panel.classList.toggle("is-active", selected);
    });
  }

  tabs.forEach(function (tab, index) {
    tab.addEventListener("click", function () { activateTab(tab); });
    tab.addEventListener("keydown", function (event) {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const direction = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + direction + tabs.length) % tabs.length];
      activateTab(next);
      next.focus();
    });
  });

  let toastTimer;
  function showToast(message) {
    toast.textContent = message;
    toast.classList.add("is-visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 1800);
  }

  const copyButton = document.getElementById("copy-bibtex");
  copyButton.addEventListener("click", async function () {
    const citation = document.getElementById("bibtex").innerText;
    try {
      await navigator.clipboard.writeText(citation);
      copyButton.querySelector("span").textContent = "Copied";
      showToast("Citation copied");
      window.setTimeout(function () {
        copyButton.querySelector("span").textContent = "Copy BibTeX";
      }, 1800);
    } catch (error) {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(document.getElementById("bibtex"));
      selection.removeAllRanges();
      selection.addRange(range);
      showToast("Citation selected — press Ctrl/Cmd+C");
    }
  });

  const dialog = document.getElementById("image-dialog");
  const dialogImage = document.getElementById("dialog-image");
  const dialogCaption = document.getElementById("dialog-caption");

  document.querySelectorAll(".figure-open").forEach(function (button) {
    button.addEventListener("click", function () {
      if (typeof dialog.showModal !== "function") {
        window.open(button.dataset.image, "_blank", "noopener");
        return;
      }
      const sourceImage = button.querySelector("img");
      const figure = button.closest("figure");
      dialogImage.src = button.dataset.image;
      dialogImage.alt = sourceImage.alt;
      dialogCaption.textContent = figure.querySelector("figcaption").textContent;
      dialog.showModal();
    });
  });

  dialog.addEventListener("click", function (event) {
    const box = dialog.getBoundingClientRect();
    const inside = event.clientX >= box.left && event.clientX <= box.right && event.clientY >= box.top && event.clientY <= box.bottom;
    if (!inside) dialog.close();
  });
})();
