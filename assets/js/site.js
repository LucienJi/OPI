// Page behaviour shared by both project pages: video placeholders, scoped
// play/pause, a synchronized two-video player, nav highlighting, BibTeX copy.

(function () {
  const IMAGE_ICON =
    '<svg class="ph-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"' +
    ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4.5" width="18" height="15" rx="2"/>' +
    '<circle cx="8.5" cy="9.5" r="1.6"/><path d="m21 16-5-5-9 9"/></svg>';

  const PLACEHOLDER_ICON =
    '<svg class="ph-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"' +
    ' stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2.5" y="5" width="14" height="14" rx="2"/>' +
    '<path d="M16.5 10.5 21.5 7.5v9l-5-3"/></svg>';

  // <video|img data-placeholder="Label"> swaps itself for a labelled drop-in slot
  // when its file is missing, showing the exact path to place the file at.
  function placeholderFor(video) {
    const box = document.createElement("div");
    box.className = "video-placeholder";
    const src = video.getAttribute("src") || video.querySelector("source")?.getAttribute("src") || "";
    const inner = document.createElement("div");
    inner.innerHTML = video.tagName === "IMG" ? IMAGE_ICON : PLACEHOLDER_ICON;
    const title = document.createElement("strong");
    title.textContent = video.dataset.placeholder || "Coming soon";
    const hint = document.createElement("div");
    hint.textContent = video.tagName === "IMG" ? "Drop the image at" : "Drop the clip at";
    const code = document.createElement("code");
    code.textContent = src;
    inner.append(title, hint, code);
    box.appendChild(inner);
    if (video.style.aspectRatio) box.style.aspectRatio = video.style.aspectRatio;
    return box;
  }

  function armPlaceholders(root = document) {
    root.querySelectorAll("img[data-placeholder]").forEach((img) => {
      if (img.dataset.armed) return;
      img.dataset.armed = "1";
      const swap = () => {
        if (!img.isConnected) return;
        const box = placeholderFor(img);
        box.classList.add("image-placeholder");
        if (img.dataset.aspect) box.style.aspectRatio = img.dataset.aspect;
        img.replaceWith(box);
      };
      img.addEventListener("error", swap);
      if (img.complete && img.naturalWidth === 0) swap();
    });
    root.querySelectorAll("video[data-placeholder]").forEach((video) => {
      if (video.dataset.armed) return;
      video.dataset.armed = "1";
      const swap = () => {
        if (!video.isConnected) return;
        video.replaceWith(placeholderFor(video));
      };
      video.addEventListener("error", swap, true);
      const source = video.querySelector("source");
      source?.addEventListener("error", swap);
      // A <source> that failed before arming leaves NETWORK_NO_SOURCE behind. With a src
      // attribute that state is also set transiently while loading starts, so only a
      // recorded error counts there.
      if (video.error || (source && video.networkState === HTMLMediaElement.NETWORK_NO_SOURCE)) swap();
    });
  }

  // [data-video-action="play|pause"][data-video-target="scope"]
  document.addEventListener("click", (event) => {
    const button = event.target instanceof Element && event.target.closest("[data-video-action]");
    if (!button) return;
    const scope = document.querySelector(`[data-video-scope="${button.dataset.videoTarget}"]`);
    const videos = Array.from((scope || document).querySelectorAll("video"));
    if (button.dataset.videoAction === "play") {
      videos.forEach((v) => {
        v.muted = true;
        try { v.currentTime = 0; } catch { /* metadata not ready */ }
      });
      Promise.allSettled(videos.map((v) => v.play()));
    } else {
      videos.forEach((v) => v.pause());
    }
  });

  // Drive two or more videos from a single transport: they start, pause, seek
  // and change rate together. The clip that runs longest owns the timeline;
  // shorter clips hold their last frame.
  function syncedPlayer(videos, controls) {
    const { play, restart, scrub, time, rate } = controls;
    let duration = 0;
    let seeking = false;

    const longest = () => videos.reduce((a, b) => ((b.duration || 0) > (a.duration || 0) ? b : a), videos[0]);
    const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
    const playing = () => videos.some((v) => !v.paused && !v.ended);

    function refreshDuration() {
      duration = Math.max(0, ...videos.map((v) => v.duration || 0));
      if (scrub) scrub.max = String(duration || 0);
    }

    function setPlayLabel() {
      if (play) play.textContent = playing() ? "Pause" : "Play";
    }

    function tick() {
      const lead = longest();
      if (lead && scrub && !seeking) scrub.value = String(lead.currentTime);
      if (lead && time) time.textContent = `${fmtTime(lead.currentTime)} / ${fmtTime(duration)}`;
      if (playing()) requestAnimationFrame(tick);
      setPlayLabel();
    }

    function seek(t) {
      videos.forEach((v) => {
        try { v.currentTime = Math.min(t, (v.duration || t) - 0.001); } catch { /* not ready */ }
      });
      tick();
    }

    async function start() {
      const lead = longest();
      if (lead && lead.ended) seek(0);
      videos.forEach((v) => { v.muted = true; });
      await Promise.allSettled(videos.filter((v) => !(v.duration && v.currentTime >= v.duration - 0.01)).map((v) => v.play()));
      requestAnimationFrame(tick);
    }

    function pause() {
      videos.forEach((v) => v.pause());
      tick();
    }

    videos.forEach((v) => {
      v.addEventListener("loadedmetadata", () => { refreshDuration(); tick(); });
      v.addEventListener("ended", tick);
      v.addEventListener("pause", setPlayLabel);
      v.addEventListener("play", setPlayLabel);
    });
    refreshDuration();

    play?.addEventListener("click", () => (playing() ? pause() : start()));
    restart?.addEventListener("click", () => { seek(0); start(); });
    scrub?.addEventListener("input", () => { seeking = true; pause(); seek(Number(scrub.value)); });
    scrub?.addEventListener("change", () => { seeking = false; });
    rate?.addEventListener("change", () => videos.forEach((v) => { v.playbackRate = Number(rate.value); }));

    return { start, pause, seek, refreshDuration };
  }

  // Highlight the nav link for the section currently in view.
  function scrollSpy() {
    const links = Array.from(document.querySelectorAll(".top-nav nav a[href^='#']"));
    const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    if (!targets.length) return;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`));
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    targets.forEach((t) => observer.observe(t));
  }

  function copyButtons() {
    document.querySelectorAll("[data-copy-target]").forEach((button) => {
      button.addEventListener("click", async () => {
        const target = document.getElementById(button.dataset.copyTarget);
        if (!target) return;
        try {
          await navigator.clipboard.writeText(target.textContent.trim());
          button.textContent = "Copied";
        } catch {
          button.textContent = "Select & copy";
        }
        setTimeout(() => { button.textContent = "Copy"; }, 1600);
      });
    });
  }

  // Lazy autoplay: muted looping clips only play while on screen.
  function lazyAutoplay(root = document) {
    const videos = root.querySelectorAll("video[data-autoplay]");
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const v = entry.target;
        if (entry.isIntersecting) {
          v.muted = true;
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      }
    }, { threshold: 0.25 });
    videos.forEach((v) => observer.observe(v));
  }

  document.addEventListener("DOMContentLoaded", () => {
    armPlaceholders();
    scrollSpy();
    copyButtons();
    lazyAutoplay();
  });

  window.Site = { armPlaceholders, syncedPlayer, lazyAutoplay };
})();
