# Optional hosting: Render + Neon + GitHub Pages

The main way to run agent-replay is `docker compose up` (see the README). This guide puts a
read-only copy online on free tiers: Postgres on Neon, the API on a Render web service and the
viewer on GitHub Pages. Every step is manual, in the provider dashboards and a local shell.

## Free-tier limits (as of November 2026)

- Render free web service: spins down after 15 minutes without traffic and takes about a minute
  to wake; 750 instance hours per workspace per month; one instance, no persistent disk
  ([render.com/docs/free](https://render.com/docs/free)). Whether a card is needed: check at
  signup.
- Render free Postgres expires 30 days after creation, so it is not used here.
- Neon free: no card, 1 GB storage per project, compute scales to zero after 5 minutes idle and
  resumes on the first query, 100 CU-hours per project per month
  ([neon.com/pricing](https://neon.com/pricing)).
- GitHub Pages: 1 GB per site, soft limit of 100 GB bandwidth per month
  ([Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)).

Both sleep rules stack, so the first request after a quiet period can wait for the API and then
for Neon. The viewer shows a "waking up" state and retries once.

## 1. Postgres on Neon

1. Create a Neon project in the region closest to the Render region you plan to use, and copy
   its connection string (`postgresql://<user>:<password>@<host>/<dbname>?sslmode=require`).
   Treat it as a secret.
2. From your machine, with that owner connection string, apply the migrations and load the
   committed fixtures so the viewer has data:

   ```bash
   export AGENT_REPLAY_DATABASE_URL='<owner connection string>'
   uv run agent-replay db migrate
   uv run agent-replay fixtures import evals/baseline
   uv run agent-replay fixtures import evals/regression
   ```

3. Create a read-only role for the API, since it never writes (Neon SQL editor or `psql`):

   ```sql
   CREATE ROLE agent_replay_reader LOGIN PASSWORD '<new password>';
   GRANT CONNECT ON DATABASE <dbname> TO agent_replay_reader;
   GRANT USAGE ON SCHEMA public TO agent_replay_reader;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_replay_reader;
   ```

   Build a second connection string with this role. Only that one goes to Render.

## 2. API on a Render web service

1. Create a Web Service from your GitHub repository on the free instance type, with either
   runtime:
   - Python 3.12: build command `pip install .`, start command
     `uvicorn agent_replay.api.asgi:app --host 0.0.0.0 --port 8000`.
   - Docker: Render builds the repository `Dockerfile`, whose default command is the same
     uvicorn command. That image also serves the viewer at `/`, so step 3 becomes optional.
2. Make sure Render routes traffic to port 8000 (check the service settings and logs after the
   first deploy).
3. Set the environment variables in the Render dashboard:

   | Name | Value |
   |---|---|
   | `AGENT_REPLAY_DATABASE_URL` | the read-only connection string from part 1 |
   | `AGENT_REPLAY_CORS_ORIGINS` | the Pages origin as a JSON list, for example `["https://<user>.github.io"]` |
   | `AGENT_REPLAY_ALLOWED_HOSTS` | the service's host name, `<service>.onrender.com` (comma-separated if you add a custom domain) |

   The origin is scheme and host only, lowercase, with no path and no trailing slash. The API
   allows GET only, sends no credentials and refuses `*`. It answers only the host names in
   `AGENT_REPLAY_ALLOWED_HOSTS` (default `localhost,127.0.0.1`); without that variable every
   request to the Render URL gets 400. No model key is needed: the API only reads runs that are
   already stored.
4. After the deploy, open `https://<service>.onrender.com/api/health`; it answers
   `{"status": "ok"}` once the service and the database are awake.

## 3. Viewer on GitHub Pages

`VITE_API_URL` is the Render URL plus `/api`, with no trailing slash. Vite inlines every `VITE_`
variable into the public bundle, so put nothing secret in one. A project site is served under
`/<repo>/`, so pass the base path at build time; a user or custom-domain site at `/` does not
need it:

```bash
cd viewer
npm ci
VITE_API_URL=https://<service>.onrender.com/api npm run build -- --base=/<repo>/
```

The result is `viewer/dist`. The viewer routes in the URL hash (`#/runs/<id>`), so no `404.html`
fallback is needed.

To publish it, set the repository's Pages source to GitHub Actions and add a workflow (this
repository has none) that runs the build above in `viewer/` with `VITE_API_URL` set, uploads
`viewer/dist` with `actions/upload-pages-artifact`, and publishes it with
`actions/deploy-pages`, with `pages: write` and `id-token: write` permissions. Pin each action to
a commit SHA, as the regression-gate workflow does.

Open `https://<user>.github.io/<repo>/`. A network or CORS error usually means
`AGENT_REPLAY_CORS_ORIGINS` does not match the Pages origin exactly, or the API is still waking
(the viewer retries once after 30 s). A 400 from the API means `AGENT_REPLAY_ALLOWED_HOSTS`
does not name the Render host.
