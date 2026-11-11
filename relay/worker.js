// OC2VRM's report relay: a Cloudflare Worker between the app's "Send to the developers" and the Discord
// channel reports go to. The webhook's address stays here, a secret (DISCORD_WEBHOOK), never in the app or
// the repository: anyone who has a webhook's address can post to it, rename it or delete it, and once the
// app is public its files are anyone's to read.
//
// It takes what the app posts (app/report.py `send`): a multipart form, "payload_json" with the message and
// "files[0]" with the report's zip. It checks that it is that and no bigger than a Discord message takes,
// lets each sender send only so often, and passes it on with no one pinged.

const MOST = Math.floor(9.5 * 1024 * 1024); // what Discord takes in one message's files, with the form round it
const NOTE_MOST = 2000; // Discord's limit for a message's text
const ZIP = [0x50, 0x4b, 0x03, 0x04]; // "PK\3\4": a zip's first bytes

function reply(status, text) {
  return new Response(JSON.stringify({ ok: status === 200, said: text }), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

async function allowed(limiter, key) {
  // (no limiter bound, as in a test: nothing is limited)
  if (!limiter) return true;
  const { success } = await limiter.limit({ key });
  return success;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/report") return reply(404, "nothing here");
    if (request.method !== "POST") return reply(405, "reports are posted");
    if (!(request.headers.get("content-type") || "").startsWith("multipart/form-data")) return reply(415, "not a report");
    const size = Number(request.headers.get("content-length") || 0);
    if (size > MOST) return reply(413, "too large: send it without pictures, or hand the zip over");

    // each sender a few a minute, and all of them together a few dozen: a flood is stopped here, not in the channel
    const sender = request.headers.get("cf-connecting-ip") || "unknown";
    if (!(await allowed(env.PER_SENDER, sender))) return reply(429, "too many just now: try again in a minute");
    if (!(await allowed(env.ALL, "all"))) return reply(429, "too many just now: try again in a minute");

    let form;
    try {
      form = await request.formData();
    } catch {
      return reply(400, "not a report");
    }
    const payload = form.get("payload_json");
    const file = form.get("files[0]");
    if (typeof payload !== "string" || !file || typeof file === "string") return reply(400, "not a report");
    let content;
    try {
      content = String(JSON.parse(payload).content ?? "");
    } catch {
      return reply(400, "not a report");
    }
    if (file.size > MOST) return reply(413, "too large: send it without pictures, or hand the zip over");
    const head = new Uint8Array(await file.slice(0, 4).arrayBuffer());
    if (!String(file.name || "").toLowerCase().endsWith(".zip") || !ZIP.every((byte, i) => head[i] === byte)) {
      return reply(400, "a report is a zip");
    }
    if (!env.DISCORD_WEBHOOK) return reply(503, "the relay has no channel to send to");

    // passed on as the app would have sent it, with no one pinged whatever the text says
    const name = String(file.name).replace(/[^\w.-]/g, "_").slice(-100);
    const out = new FormData();
    out.append("payload_json", JSON.stringify({ content: content.slice(0, NOTE_MOST), allowed_mentions: { parse: [] } }));
    out.append("files[0]", new File([await file.arrayBuffer()], name, { type: "application/zip" }));
    let sent;
    try {
      sent = await fetch(env.DISCORD_WEBHOOK, { method: "POST", body: out });
    } catch {
      return reply(502, "the channel couldn't be reached: try again later");
    }
    if (sent.status === 429) return reply(429, "too many just now: try again in a minute");
    if (!sent.ok) return reply(502, `the channel refused it (${sent.status})`);
    return reply(200, "sent");
  },
};
