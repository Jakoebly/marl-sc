import wandb
api = wandb.Api()

ids = ["a3658ab0"] 
for rid in ids:
    path = f"gowiththetensorflow/marl-sc/{rid}"
    try:
        state = api.run(path).state
    except Exception as e:
        print(f"{rid}: not found ({e})"); continue
    print(f"{rid}: {state}")
    if state == "finished":
        print(f"{rid}: already finished, skipping"); continue
    wandb.init(project="marl-sc", entity="gowiththetensorflow", id=rid, resume="must")
    wandb.finish()
    print(f"{rid}: was {state} -> finished")


