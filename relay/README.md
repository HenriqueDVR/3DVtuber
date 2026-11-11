# Report relay

The app's **Send to the developers** posts a report (a message and a zip) to the address in
`app/reports.json`. Once the repository is public, anything in it can be read by anyone. A Discord
webhook's address lets whoever has it post to the channel, rename the webhook or delete it. So the app
posts to this relay instead: a Cloudflare Worker that keeps the webhook as a secret and passes reports on.

What it does (`worker.js`):

- takes only `POST /report`, a multipart form with the message (`payload_json`) and one `.zip` that really is a zip,
  9.5 MB at most;
- lets each sender send 3 a minute, and everyone together 30 a minute (`wrangler.toml`);
- passes it on with no one pinged (`allowed_mentions`), and never shows the webhook's address in an answer.

`node --test relay/test.mjs` runs it against a stand-in for Discord (also part of `python -m unittest tests.test_app`).

## Setting it up (once)

1. **A Cloudflare account**, free: <https://dash.cloudflare.com/sign-up>.
2. **A new webhook in Discord**: Server Settings > Integrations > Webhooks. Delete the old one (its address is
   in the repository's history), then New Webhook, pick the reports channel, **Copy Webhook URL**. Keep it to paste in step 5.
3. Open a terminal in this folder (`relay`).
4. Sign in and put the relay online (the first run asks to download Wrangler, Cloudflare's tool, and opens the browser to sign in):

   ```bash
   npx wrangler@4 login
   ```

   ```bash
   npx wrangler@4 deploy
   ```

   The first deploy asks for a `workers.dev` name of your own. It ends by printing the relay's address:
   `https://oc2vrm-reports.<your name>.workers.dev`.
5. Give the relay the webhook (paste the address from step 2 when asked):

   ```bash
   npx wrangler@4 secret put DISCORD_WEBHOOK
   ```

6. Put the relay's address, with `/report` on the end, in `app/reports.json` as `"address"`, and send a report from
   the app to try it.

To see what it does live: `npx wrangler@4 tail`. To change the limits, edit `wrangler.toml` and deploy again.
