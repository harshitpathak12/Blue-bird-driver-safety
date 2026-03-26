# MongoDB – Why data might not appear

Use this when the app runs and external features work, but nothing shows up in MongoDB.

## 1. Confirm database name

The app uses the database name from **`configs/config.yaml`**:

- **`database: "driver_monitoring"`**

In MongoDB Compass (or Atlas UI), make sure you are viewing the **`driver_monitoring`** database, not another one (e.g. `test`, `admin`).

## 2. Startup connection check

On startup the app now:

- Connects with a 5s timeout
- Pings the server
- Lists collections in the configured database

- **If connection fails:** the process will **exit with an error** (e.g. `MongoDB connection failed: ...`). Fix the URL, network, or Atlas IP access and restart.
- **If you see:** `MongoDB connected: database=driver_monitoring` → connection and database name are correct.

## 3. Where data is written

| Collection   | When it gets documents |
|-------------|-------------------------|
| **alerts**  | WebSocket or REST: only when the fusion engine raises an **alert** (fatigue / distraction / sleep). |
| **sessions**| When the first alert occurs for a connection and a `driver_id` is available (from URL or face recognition), or when `POST /api/sessions/start` is used. |
| **drivers** | When `POST /api/login/register` is used. |

So:

- **No alerts:** Normal if the driver is not in a fatigue/distraction/sleep state.
- **No sessions:** Normal if no alert has happened yet or no `driver_id` is set/recognized.

## 4. `driver_id` and “sessions” events

- If the client does **not** pass `?driver_id=...` and face recognition does **not** identify a driver, alerts use driver_id "UNKNOWN" and no session is created.
## 5. Write errors in the console

If an insert fails (e.g. permissions, schema, network), the app will log:

- `[alert_repository] insert_one failed: ...`
- `[session_repository] insert_one failed: ...`

and the WebSocket handler may log:

- `WebSocket error: ...`

Check the **server console** for these messages when testing the stream or creating alerts/sessions.

## 6. Run the correct app

Use the main app that includes both REST APIs and WebSocket and uses the same config:

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 5000
```

If you run another app (e.g. a minimal one that only mounts the WebSocket), it may still use the same `configs/config.yaml` and MongoDB, but ensure that app actually imports and uses the same `database` and `configs` packages.

## 7. Config file location

The config path is resolved from the **package directory** (absolute path), so it does not depend on the current working directory. If the server fails at startup with a missing config file, fix the path or run from the project root so that the `configs` package is importable.

## 8. Atlas / network

- **Atlas:** IP Access List must allow your server (or `0.0.0.0/0` for testing).
- **URL:** In `configs/config.yaml`, `mongodb.url` must be correct (including username/password and cluster host).
- **Network:** Server must be able to reach the MongoDB host (no firewall blocking outbound traffic to the Mongo port).

After changing config or network, restart the app and watch for the startup message and any `insert_one failed` or `WebSocket error` logs.
