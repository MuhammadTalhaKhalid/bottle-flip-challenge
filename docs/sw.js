/* sw.js — PWA caching tuned so the client never has to clear cache.
 *
 * App shell (HTML/CSS/JS/icons) is served NETWORK-FIRST: opening the link on a
 * phone always fetches the latest deploy, falling back to cache only when
 * offline. The large, rarely-changing model files (*.onnx, ~27 MB) stay
 * CACHE-FIRST so they load instantly and work offline after the first visit.
 */
const CACHE = "bottleflip-v8";
const ASSETS = [
  "./", "./index.html", "./styles.css", "./app.js", "./engine.js", "./yolo.js",
  "./manifest.webmanifest",
  "./icons/icon-192.png", "./icons/icon-512.png",
  "./models/yolov8n.onnx", "./models/flip_classifier.onnx",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  // Let cross-origin requests (e.g. the onnxruntime CDN) go straight to network.
  if (e.request.method !== "GET" || url.origin !== self.location.origin) return;

  // Big immutable model files: cache-first (instant + offline), backfill on miss.
  if (url.pathname.endsWith(".onnx")) {
    e.respondWith(
      caches.match(e.request).then((r) => r || fetch(e.request).then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      }))
    );
    return;
  }

  // App shell: network-first so a new deploy shows up on the next open; fall back
  // to the cached copy (or index.html for navigations) when offline.
  e.respondWith(
    fetch(e.request).then((res) => {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(e.request, copy));
      return res;
    }).catch(() => caches.match(e.request).then((r) => r || caches.match("./index.html")))
  );
});
