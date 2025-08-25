# Collection of different Machine Learning (ML) and Exploratory Data Analysis (EDA) projects from public datasets

## Dash Dashboard (Spotify)
A Dash dashboard is available for the Spotify dataset.

How to run:
- Ensure dependencies are installed (using uv/pip):
  - With uv: `uv sync`
  - Or with pip: `pip install -e .`
- Start the app:
  - `python -m Spotify.dashboard`
  - or `python Spotify\dashboard.py`
- Open http://127.0.0.1:8050/ in your browser.

Data file path: `Spotify\data\spotify-2023.csv`. The app loads this file with encoding ISO-8859-2.
