// Serves tinydublin.com from the R2 bucket bound as MAP.
export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (url.hostname === "www.tinydublin.com") {
      url.hostname = "tinydublin.com";
      return Response.redirect(url.toString(), 301);
    }
    if (req.method !== "GET" && req.method !== "HEAD") return new Response("Method not allowed", { status: 405 });
    let key = decodeURIComponent(url.pathname.slice(1));
    if (key === "" || key.endsWith("/")) key += "index.html";
    const isTile = key.startsWith("tiles/map_files/");
    const cache = caches.default;
    const cacheKey = new Request(url.origin + "/" + key, { method: "GET" });
    if (isTile) {
      const hit = await cache.match(cacheKey);
      if (hit) return req.method === "HEAD" ? new Response(null, hit) : hit;
    }
    const obj = await env.MAP.get(key);
    if (!obj) return new Response("Not found", { status: 404 });
    const headers = new Headers();
    obj.writeHttpMetadata(headers);
    headers.set("etag", obj.httpEtag);
    headers.set("access-control-allow-origin", "*");
    headers.set("cache-control", isTile ? "public, max-age=31536000, immutable" : "public, max-age=60");
    const res = new Response(req.method === "HEAD" ? null : obj.body, { headers });
    if (isTile && req.method === "GET") ctx.waitUntil(cache.put(cacheKey, res.clone()));
    return res;
  },
};
