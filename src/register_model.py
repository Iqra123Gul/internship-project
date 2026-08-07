import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("flickr8k-blip-finetuning")

MODEL_DIR = os.environ.get("MODEL_DIR", "models/week2_baseline_blip")
MODEL_NAME = "flickr8k-blip-baseline"


def register_baseline_model():
    client = MlflowClient()

    with mlflow.start_run(run_name="register_week2_baseline") as run:
        mlflow.log_params(
            {
                "learning_rate": 5e-5,
                "subset_size": 1000,
                "decoding": "beam_search",
                "num_beams": 3,
            }
        )
        mlflow.log_metrics({"bleu": 0.093, "rougeL": 0.397, "train_loss": 2.77})

        mlflow.log_artifacts(MODEL_DIR, artifact_path="blip_baseline")

        run_id = run.info.run_id
        artifact_uri = f"runs:/{run_id}/blip_baseline"

        # Ensure the registered model exists (create if not already there)
        try:
            client.create_registered_model(MODEL_NAME)
        except mlflow.exceptions.MlflowException:
            pass  # already exists, that's fine

        # Create a new version pointing directly to the artifact
        model_version = client.create_model_version(
            name=MODEL_NAME, source=artifact_uri, run_id=run_id
        )

        print(f"Model registered: {MODEL_NAME}")
        print(f"Run ID: {run_id}")
        print(f"Artifact URI: {artifact_uri}")
        print(f"Registered Version: {model_version.version}")


if __name__ == "__main__":
    register_baseline_model()
