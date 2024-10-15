import wandb
import time

PROJECT = "Domain-Game"


def init_wandb(config):

    wandb.login(key=config['wandb_key'])
    model_name = f"{config['experiment']}"

    run = wandb.init(
        # Set the project where this run will be logged
        project=PROJECT,
        tags="Benchmark",
        name=model_name,

        # Track hyperparameters and run metadata
        config=config
    )




# Put the log
# wandb.log({'Train Loss': batch_loss,
#            'Val Accuracy': acc})

# wandb.log({**metrics, **val_metrics})
# metrics is a dict

# If you had a test set, this is how you could log it as a Summary metric
# wandb.summary['test_accuracy'] = 0.8

# wandb.finish()



# Log one batch of images to the dashboard, always same batch_idx.
# if i==batch_idx and log_images:
#     log_image_table(images, predicted, labels, outputs.softmax(dim=1))

def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())

    wandb.log({"predictions_table":table}, commit=False)

