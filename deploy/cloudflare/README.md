# Publishing Nivra through Cloudflare Tunnel

The Raspberry Pi already runs a **remotely managed** tunnel (`cloudflared.service`, token based,
4 healthy edge connections at the time of writing). Public hostnames for such tunnels are managed
in the Cloudflare dashboard, not in a local `config.yml`. No router port forwarding is needed and
the controller keeps listening on `127.0.0.1:8740` only.

> Before publishing anything: change the Pi account password (it appeared in plain text during
> setup), create a strong Nivra administrator password, and make sure the controller runs
> permanently (see `deploy/systemd/`). If the controller is stopped, the public sites are offline.

## 1. Add public hostnames (owner, in the dashboard)

Cloudflare Zero Trust → Networks → Tunnels → *(the Pi tunnel)* → Public hostnames → Add:

| Subdomain | Domain | Service | Purpose |
|-----------|--------|---------|---------|
| `api` | `benjiserver.com` | `http://127.0.0.1:8740` | Worker protocol + API |
| `studio` | `benjiserver.com` | `http://127.0.0.1:8740` | Nivra Studio (admin) |

Check first that neither hostname already has a DNS record — adding a public hostname creates a
proxied CNAME and must not overwrite an existing record.

Optional `workers.benjiserver.com` (onboarding docs) can point to the same service; `assets.` is
not needed because artifacts are served by the API with Range support.

## 2. Controller settings for HTTPS behind the tunnel

In `config/controller.json` on the Pi:

```json
{
  "public_url": "https://api.benjiserver.com",
  "cookie_secure": true,
  "trust_proxy_headers": true
}
```

`trust_proxy_headers` makes rate limits use `CF-Connecting-IP`, and it is only honoured for
requests arriving from loopback (i.e. from `cloudflared`). Restart the controller afterwards.

## 3. Access separation

* `studio.benjiserver.com` — recommended: a Cloudflare Access application (email one-time PIN or
  SSO) **in addition to** Nivra's own login. Access protects the admin surface from the internet;
  Nivra still enforces roles and CSRF.
* `api.benjiserver.com` — **no** Cloudflare Access policy (or a bypass policy), because workers
  authenticate with their own Ed25519 device keys and cannot complete interactive Access logins.
  Never hand out Cloudflare service tokens or administrator credentials to contributors.

## 4. Verify from outside the LAN

```bash
curl -fsS https://api.benjiserver.com/api/v1/health
```

Then enroll a worker with `https://api.benjiserver.com` as the controller URL. The worker refuses
plain HTTP to public hosts.

## 5. Health in Nivra Studio

The controller reads `cloudflared`'s local metrics endpoint (`http://127.0.0.1:20241/ready`, set as
`cloudflared_metrics_url` by the installer) and shows the tunnel status on the Overview page.
