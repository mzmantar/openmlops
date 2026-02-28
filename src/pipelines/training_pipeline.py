from zenml import pipeline
from src.steps.training.ingest_data import ingest_data
from src.steps.training.upload_data_to_minio import upload_data_to_minio
from src.steps.training.validate_data import validate_data
from src.steps.training.split_data import split_data
from src.steps.training.preprocess import preprocess
from src.steps.training.train import train
from src.steps.training.evaluate import evaluate
from src.steps.training.register_model import register_model
from src.steps.training.export_model import export_model

@pipeline(enable_cache=False)
def training_pipeline():
    cifar_dir = ingest_data()
    _ = upload_data_to_minio(cifar_dir)
    cifar_dir = validate_data(cifar_dir)
    split_out = split_data(cifar_dir)
    preprocess_out = preprocess(split_out)
    trained_model = train(preprocess_out)
    _ = evaluate(trained_model, preprocess_out)
    _ = register_model(trained_model)
    _ = export_model(trained_model)

if __name__ == "__main__":
    training_pipeline()  # ZenML will run it