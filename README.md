Integrador - Visualizaciones
============================

This repository now includes reproducible Altair visuals for the Mendoza real-estate dataset processed in the previous deliveries (`include/data/processed/propiedades_clean.csv`). The entrypoint is `visualizations/altair_charts.py`, which exposes reusable helpers and a CLI to materialise the charts that will later feed the Streamlit app.

Usage
-----

1. Create a virtual environment and install the Python requirements:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Generate the interactive charts (HTML files are emitted under `visualizations/charts/`):

   ```bash
   python -m visualizations.altair_charts
   ```

   The module creates three linked outputs:

   - `price_distribution.html`: box plots (log scale) that compare USD listing prices by property type and operation (Venta, Alquiler, etc.).
   - `price_vs_surface.html`: scatter plot on log-log axes that relates total area (m2) and price, with legend-driven filtering by property type and point size tied to bedrooms.
   - `price_map.html`: geographic view centred on Gran Mendoza that encodes price intensity via circle size and exposes locality, operation and surface in the tooltip.

   All charts are built on top of the `load_properties_data` helper, which normalises datatypes (dates, numerics and geospatial coordinates) and computes a consistent set of derived fields (`precio_usd`, `superficie_total`, `cocheras_total`, among others). These functions are designed to be imported directly in the future Streamlit app.

3. (Optional) Use the returned dictionary from `save_charts()` to embed the specs elsewhere in the project.

> Note: If Altair is not installed you will receive a descriptive runtime error. Run `pip install altair` or the steps above to fix it.

Streamlit App
-------------

Once the dependencies are installed you can launch the interactive dashboard with:

```bash
streamlit run streamlit_app.py
```

The app includes three coordinated sections:

- **Resumen**: key dataset metrics, validation scores of the linear baseline (MAE, RMSE, R2) and a preview of the filtered listings.
- **Visualizaciones**: embeds the Altair charts directly inside Streamlit, keeping the same interactivity (filters are propagated from the sidebar).
- **Modelo**: form to enter a new property (operacion, tipo, localidad, metros, ambientes, etc.) and obtain an instantaneous USD price estimate using the fitted linear model (log target, z-score scaling, one-hot encoding).

The model artefacts are built on demand with `app/model.py`. It applies simple imputations, standardisation and a least-squares solver on the log of the price. The current baseline prioritises transparency over accuracy; large errors on outliers are expected and are tracked via the metrics shown in the UI.

Overview
========

Welcome to Astronomer! This project was generated after you ran 'astro dev init' using the Astronomer CLI. This readme describes the contents of the project, as well as how to run Apache Airflow on your local machine.

Project Contents
================

Your Astro project contains the following files and folders:

- dags: This folder contains the Python files for your Airflow DAGs. By default, this directory includes one example DAG:
    - `example_astronauts`: This DAG shows a simple ETL pipeline example that queries the list of astronauts currently in space from the Open Notify API and prints a statement for each astronaut. The DAG uses the TaskFlow API to define tasks in Python, and dynamic task mapping to dynamically print a statement for each astronaut. For more on how this DAG works, see our [Getting started tutorial](https://www.astronomer.io/docs/learn/get-started-with-airflow).
- Dockerfile: This file contains a versioned Astro Runtime Docker image that provides a differentiated Airflow experience. If you want to execute other commands or overrides at runtime, specify them here.
- include: This folder contains any additional files that you want to include as part of your project. It is empty by default.
- packages.txt: Install OS-level packages needed for your project by adding them to this file. It is empty by default.
- requirements.txt: Install Python packages needed for your project by adding them to this file. It is empty by default.
- plugins: Add custom or community plugins for your project to this file. It is empty by default.
- airflow_settings.yaml: Use this local-only file to specify Airflow Connections, Variables, and Pools instead of entering them in the Airflow UI as you develop DAGs in this project.

Deploy Your Project Locally
===========================

Start Airflow on your local machine by running 'astro dev start'.

This command will spin up five Docker containers on your machine, each for a different Airflow component:

- Postgres: Airflow's Metadata Database
- Scheduler: The Airflow component responsible for monitoring and triggering tasks
- DAG Processor: The Airflow component responsible for parsing DAGs
- API Server: The Airflow component responsible for serving the Airflow UI and API
- Triggerer: The Airflow component responsible for triggering deferred tasks

When all five containers are ready the command will open the browser to the Airflow UI at http://localhost:8080/. You should also be able to access your Postgres Database at 'localhost:5432/postgres' with username 'postgres' and password 'postgres'.

Note: If you already have either of the above ports allocated, you can either [stop your existing Docker containers or change the port](https://www.astronomer.io/docs/astro/cli/troubleshoot-locally#ports-are-not-available-for-my-local-airflow-webserver).

Deploy Your Project to Astronomer
=================================

If you have an Astronomer account, pushing code to a Deployment on Astronomer is simple. For deploying instructions, refer to Astronomer documentation: https://www.astronomer.io/docs/astro/deploy-code/

Contact
=======

The Astronomer CLI is maintained with love by the Astronomer team. To report a bug or suggest a change, reach out to our support.
