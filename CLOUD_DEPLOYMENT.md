# Cloud deployment

The app is packaged for Streamlit Community Cloud. It includes the trained TXL-PBC checkpoint at `models/blood_cell_counter.pt`, so it can count uploaded images without the training dataset or local `runs/` folder.

## Deploy with Streamlit Community Cloud

1. Push this project to a GitHub repository that your Streamlit Community Cloud account can access. Include `main.py`, `requirements.txt`, `.streamlit/config.toml`, and `models/blood_cell_counter.pt`.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and connect the GitHub account that can access the repository.
3. Choose **Create app**, select the repository and branch, and set the app file to `main.py`.
4. In advanced settings, select Python 3.12, then deploy.

Streamlit Community Cloud builds the app from the selected GitHub repo and entrypoint. The uploaded image is processed in the app session and is not written to a file or database by this project. A deployed app link can be opened by anyone who can access that app; keep the repository and app private if your hosting account supports the access controls you need.

## Run the Docker image

From the project root:

```bash
docker build -t blood-smear-counter .
docker run --rm -p 8501:8501 blood-smear-counter
```

Then open `http://localhost:8501`. For a remote container host, publish the image and configure the service to listen on port `8501`.

## Model scope

The bundled model was trained and evaluated on TXL-PBC, not on microscope images supplied by a specific user. Cloud hosting does not change that training domain. Its results are a research/demo aid and are not intended for clinical diagnosis.
