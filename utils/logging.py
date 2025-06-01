from kaggle_secrets import UserSecretsClient
import os
import wandb


def init_wandb(project_name="geology-forecast-challenge", config=None):
    try:
        user_secrets = UserSecretsClient()
        
        wandb_api_key = user_secrets.get_secret("wandb")
        os.environ['WANDB_API_KEY'] = wandb_api_key
        
        wandb.login(key=wandb_api_key)
        
        run = wandb.init(
            project=project_name,
            config=config,
            tags=["LSTM", "Geology Forecast Challenge"],
        )
        
        print("W&B successfully initialized")
        return run
    
    except Exception as e:
        print(f"Error initializing W&B: {str(e)}")
        return None