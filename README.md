 # uber_demand_forecasting

uber-demand-prediction
==============================

Predicting demand for cabs across New York City for the next time intervals.

Local-first setup
------------

This repository now runs fully locally. The app no longer depends on cloud services or a remote model registry.

When you start the app for the first time, it bootstraps small local sample assets into `data/` and `models/` so the demo can run even if the original tracked files are not available.

Run locally with:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run tests with:

```bash
pip install -r requirements-dev.txt
pytest
```

Cloud deployment
------------

Recommended free option: Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and create a new app from your repository.
3. Select `app.py` as the entrypoint file.
4. In Advanced settings, keep Python `3.12` unless you need a different supported version.

Alternative free option: Render

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the repository.
3. If you use the included `render.yaml`, Render can pick up the build and start commands automatically.

Why not Vercel or Netlify?

This project is a long-running Python Streamlit server. Vercel and Netlify are a much better fit for static sites and serverless functions than for a persistent Streamlit app process.

Heroku
------------

This repository is now set up for Heroku with:

- `Procfile` for the Streamlit web process
- `.python-version` set to `3.12`
- `app.json` for app metadata
- `.slugignore` to keep the deploy slug smaller

Deploy steps:

```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

Useful commands:

```bash
heroku logs --tail --app your-app-name
heroku ps --app your-app-name
```
