// The relay, run as Cloudflare runs it (a fetch handler), against a stand-in for Discord: node --test relay/test.mjs
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import relay from "./worker.js";

const ZIP = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 1, 2, 3, 4]);

function discord(status = 204) {
  const got = [];
  const server = createServer((request, response) => {
    const parts = [];
    request.on("data", (part) => parts.push(part));
    request.on("end", () => {
      got.push({ type: request.headers["content-type"], body: Buffer.concat(parts) });
      response.writeHead(status).end();
    });
  });
  return new Promise((done) => server.listen(0, "127.0.0.1", () => done({ server, got, url: `http://127.0.0.1:${server.address().port}/hook/secret` })));
}

function report({ content = "it broke", name = "report.zip", bytes = ZIP, method = "POST", path = "/report", ip = "1.2.3.4" } = {}) {
  const form = new FormData();
  form.append("payload_json", JSON.stringify({ content, allowed_mentions: { parse: ["everyone"] } }));
  form.append("files[0]", new File([bytes], name, { type: "application/zip" }));
  const init = { method, headers: { "cf-connecting-ip": ip } };
  if (method === "POST") init.body = form;
  return new Request(`https://oc2vrm-reports.someone.workers.dev${path}`, init);
}

test("a report is passed on to the channel, no one pinged, and the channel's address never shown", async () => {
  const { server, got, url } = await discord();
  try {
    const answer = await relay.fetch(report({ content: "@everyone it broke" }), { DISCORD_WEBHOOK: url });
    assert.equal(answer.status, 200);
    const text = await answer.text();
    assert.ok(!text.includes("secret") && !text.includes("127.0.0.1"));
    assert.equal(got.length, 1);
    assert.match(got[0].type, /^multipart\/form-data/);
    const body = got[0].body.toString("latin1");
    assert.ok(body.includes('"allowed_mentions":{"parse":[]}'));
    assert.ok(body.includes("@everyone it broke"));
    assert.ok(body.includes('filename="report.zip"'));
    assert.ok(body.includes("PK\x03\x04"));
  } finally {
    server.close();
  }
});

test("what isn't a report is turned away and nothing is sent", async () => {
  const { server, got, url } = await discord();
  const env = { DISCORD_WEBHOOK: url };
  try {
    assert.equal((await relay.fetch(report({ path: "/" }), env)).status, 404);
    assert.equal((await relay.fetch(report({ method: "GET" }), env)).status, 405);
    assert.equal((await relay.fetch(report({ bytes: new Uint8Array([1, 2, 3, 4, 5]) }), env)).status, 400);
    assert.equal((await relay.fetch(report({ name: "report.exe" }), env)).status, 400);
    const big = new Uint8Array(10 * 1024 * 1024);
    big.set([0x50, 0x4b, 0x03, 0x04]);
    assert.equal((await relay.fetch(report({ bytes: big }), env)).status, 413);
    const plain = new Request("https://x.workers.dev/report", { method: "POST", body: "hello", headers: { "content-type": "text/plain" } });
    assert.equal((await relay.fetch(plain, env)).status, 415);
    assert.equal(got.length, 0);
  } finally {
    server.close();
  }
});

test("a sender over the limit is told to wait, and the channel's own refusal is said", async () => {
  const { server, got, url } = await discord();
  const counts = {};
  const limiter = (most) => ({ async limit({ key }) { counts[key] = (counts[key] || 0) + 1; return { success: counts[key] <= most }; } });
  const env = { DISCORD_WEBHOOK: url, PER_SENDER: limiter(2), ALL: limiter(100) };
  try {
    assert.equal((await relay.fetch(report(), env)).status, 200);
    assert.equal((await relay.fetch(report(), env)).status, 200);
    assert.equal((await relay.fetch(report(), env)).status, 429);
    assert.equal((await relay.fetch(report({ ip: "5.6.7.8" }), env)).status, 200);
    assert.equal(got.length, 3);
  } finally {
    server.close();
  }
  const busy = await discord(429);
  try {
    assert.equal((await relay.fetch(report(), { DISCORD_WEBHOOK: busy.url })).status, 429);
  } finally {
    busy.server.close();
  }
  const gone = await discord(404);
  try {
    assert.equal((await relay.fetch(report(), { DISCORD_WEBHOOK: gone.url })).status, 502);
  } finally {
    gone.server.close();
  }
  assert.equal((await relay.fetch(report(), {})).status, 503);
});
