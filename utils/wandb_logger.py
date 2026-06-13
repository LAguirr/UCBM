import os
import wandb

def log_experiment_to_wandb(project_name, run_name, data_dict, api_key=None):
    """
    Logs experiment results to Weights & Biases.
    """
    # 1. Login with API key if available
    if api_key:
        wandb.login(key=api_key)
    elif os.getenv("WANDB_API_KEY"):
        wandb.login(key=os.getenv("WANDB_API_KEY"))
    else:
        # Fallback to interactive or standard login
        wandb.login()

    # 2. Initialize a new run
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=data_dict
    )

    # 3. Log the metrics
    wandb.log(data_dict)

    # 4. Finish the run
    wandb.finish()
    print(f"✅ Experiment '{run_name}' synced to W&B Dashboard!")
