import wandb
import yaml
PROJECT_WANDB = "tesis_uni"
ENTITY = "ml_projects"

from dotenv import load_dotenv
import os
import wandb
load_dotenv()

os.environ["WANDB_API_KEY"] =  os.getenv("WANDB_API_KEY")

def run_sweep():
    # Cargar la configuración desde el archivo config.yaml
    with open('config_face_detection.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Inicializar wandb
    wandb.init(config=config)

    # Ejecutar el sweep
    sweep_id = wandb.sweep(config, project=PROJECT_WANDB, entity=ENTITY)

    # Obtener el enlace al tablero de wandb
    sweep_url = f'https://wandb.ai/{ENTITY}/{PROJECT_WANDB}/sweeps/{sweep_id}'
    print(f'Sweep URL: {sweep_url}')

if __name__ == "__main__":
    run_sweep()
