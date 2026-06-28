/* sw.js — cache the app shell + models so it installs as a PWA and runs offline
 * after the first load. Bump CACHE to invalidate when files change.
 */
const CACHE = "bottleflip-v2";   // ← Force browser to reload new files
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
  // network-first for the CDN (onnxruntime), cache-first for our own assets
  if (url.origin === self.location.origin) {
    e.respondWith(caches.match(e.request).then((r) => r || fetch(e.request)));
  }
});
