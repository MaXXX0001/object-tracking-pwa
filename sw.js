const CACHE = "tracking-v4";
const SHELL = ["./app.js", "./tracker.js", "./manifest.webmanifest", "./models/yolo11n.onnx"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((name) => name !== CACHE).map((name) => caches.delete(name)))),
  );
  self.clients.claim();
});

function fetchAndCache(request) {
  return fetch(request).then((response) => {
    if (response.ok && new URL(request.url).origin === location.origin) {
      const copy = response.clone();
      caches.open(CACHE).then((cache) => cache.put(request, copy));
    }
    return response;
  });
}

// Big static files are cache-first; app code is network-first so updates show up, with the cache as offline fallback
self.addEventListener("fetch", (event) => {
  const request = event.request;
  // Navigations go straight to the browser: a worker fetch cannot show the self-signed certificate warning, so it would fail or serve a stale page
  if (request.mode === "navigate" || request.method !== "GET" || request.headers.has("range")) return;
  const isHeavy = /\/(vendor|models)\//.test(new URL(request.url).pathname);
  if (isHeavy) {
    event.respondWith(caches.match(request).then((cached) => cached || fetchAndCache(request)));
    return;
  }
  event.respondWith(fetchAndCache(request).catch(() => caches.match(request)));
});
