############################################################
# Trigait Terrain Regression Training Framework
#
# config.yaml driven version
#
# Input:
#   IMU [B,200,42]
#
# Output:
#   terrain regression [B,30,6]
#   terrain classification heads
#
############################################################


import os
import sys
import argparse
import random
import shutil
import yaml


import numpy as np
import pandas as pd


import torch
import torch.nn.functional as F


from torch.utils.data import DataLoader, WeightedRandomSampler



############################################################
# Path
############################################################


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)



sys.path.append(
    os.path.join(
        PROJECT_DIR,
        "datasets"
    )
)


sys.path.append(
    os.path.join(
        PROJECT_DIR,
        "models"
    )
)



from dataset_terrain import TrigaitTerrainDataset





############################################################
# Config
############################################################


def load_config(path):

    with open(path, "r") as f:

        cfg = yaml.safe_load(f)


    return cfg





############################################################
# Random seed
############################################################


def set_seed(seed):


    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    torch.cuda.manual_seed_all(seed)






############################################################
# Model Builder
############################################################


def build_model(
        name,
        cfg
):


    name = name.lower()



    # Handle models such as Persistence that do not define independent parameters.

    model_cfg = cfg.get(
        name.upper(),
        {}
    )




    ########################################################
    # TCN
    ########################################################


    if name == "tcn":


        from tcn_terrain import TCNTerrainRegression



        return TCNTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            hidden_channels =
            model_cfg.get(
                "hidden_channels",
                [64,128,128]
            ),


            output_len =
            cfg["prediction_length"],


            dropout =
            model_cfg.get(
                "dropout",
                0.2
            )

        )







    ########################################################
    # LSTM
    ########################################################


    elif name == "lstm":


        from lstm_terrain import LSTMTerrainRegression



        return LSTMTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            hidden_dim =
            model_cfg.get(
                "hidden_dim",
                128
            ),


            num_layers =
            model_cfg.get(
                "num_layers",
                2
            ),


            pred_len =
            cfg["prediction_length"],


            dropout =
            model_cfg.get(
                "dropout",
                0.2
            )


        )







    ########################################################
    # Transformer
    ########################################################


    elif name == "transformer":


        from transformer_terrain import TransformerTerrainRegression



        return TransformerTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            d_model =
            model_cfg.get(
                "d_model",
                128
            ),


            nhead =
            model_cfg.get(
                "n_heads",
                4
            ),


            num_layers =
            model_cfg.get(
                "num_layers",
                3
            ),


            dim_feedforward =
            model_cfg.get(
                "dim_feedforward",
                256
            ),


            output_len =
            cfg["prediction_length"],


            dropout =
            model_cfg.get(
                "dropout",
                0.2
            )

        )







    ########################################################
    # GPT4TS
    ########################################################


    elif name == "gpt4ts":


        from gpt4ts_terrain import GPT4TSTerrainRegression



        return GPT4TSTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            pred_len =
            cfg["prediction_length"],


            freeze =
            model_cfg.get(
                "freeze",
                True
            )

        )







    ########################################################
    # TimesNet
    ########################################################


    elif name == "timesnet":


        from timesnet_terrain import TimesNetTerrainRegression



        return TimesNetTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            pred_len =
            cfg["prediction_length"],


            d_model =
            model_cfg.get(
                "d_model",
                64
            ),


            d_ff =
            model_cfg.get(
                "d_ff",
                128
            ),


            num_layers =
            model_cfg.get(
                "num_layers",
                2
            ),


            dropout =
            model_cfg.get(
                "dropout",
                0.1
            )

        )







    ########################################################
    # PatchTST
    ########################################################


    elif name == "patchtst":


        from patchtst_terrain import PatchTSTTerrainRegression



        return PatchTSTTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            input_len =
            cfg["input_length"],


            pred_len =
            cfg["prediction_length"],


            patch_len =
            model_cfg.get(
                "patch_len",
                16
            ),


            stride =
            model_cfg.get(
                "stride",
                8
            )

        )







    ########################################################
    # TimeMixer
    ########################################################


    elif name == "timemixer":


        from timemixer_terrain import TimeMixerTerrainRegression



        return TimeMixerTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            pred_len =
            cfg["prediction_length"],


            d_model =
            model_cfg.get(
                "d_model",
                64
            )

        )







    ########################################################
    # Mamba
    ########################################################


    elif name == "mamba":


        from mamba_terrain import MambaTerrainRegression



        return MambaTerrainRegression(


            input_dim =
            cfg["imu_channels"],


            pred_len =
            cfg["prediction_length"],


            d_model =
            model_cfg.get(
                "d_model",
                64
            )

        )







    ########################################################
    # Persistence
    ########################################################


    elif name == "persistence":


        from persistence_terrain import PersistenceTerrainRegression


        return PersistenceTerrainRegression()






    else:


        raise ValueError(

            f"Unknown model: {name}"

        )
############################################################
# Terrain Loss
############################################################



def terrain_loss(
        pred,
        target,
        mask
):
    """
    Terrain-aware loss

    1. Ignore unconstrained dimensions through terrain_mask
    2. Increase z-direction contribution because z terrain samples
       are much fewer than xy samples

    terrain order:
    0: ipsi_x
    1: ipsi_y
    2: ipsi_z
    3: contra_x
    4: contra_y
    5: contra_z
    """

    channels = pred.shape[-1]

    # default weight for x/y
    # z directions receive higher weight
    z_weight = 3.0

    channel_weights = []

    for i in range(channels):
        if i in [2, 5]:
            channel_weights.append(z_weight)
        else:
            channel_weights.append(1.0)

    loss_total = 0.0
    weight_total = 0.0

    for i in range(channels):

        p = pred[:, :, i]
        t = target[:, :, i]
        m = mask[:, :, i]

        valid = m > 0

        if valid.sum() > 0:

            loss_i = F.mse_loss(
                p[valid],
                t[valid]
            )

            w = channel_weights[i]

            loss_total += w * loss_i
            weight_total += w

    if weight_total == 0:
        return pred.sum() * 0.0


    return loss_total / weight_total




############################################################
# Terrain Classification Loss
############################################################


############################################################
# Model forward wrapper
#
# Support:
#   new model:
#       model(imu)
#
#   old model:
#       model(imu)
#
############################################################

def model_forward(model, imu):
    return model(imu)







############################################################
# Terrain Balanced Sampler
############################################################

def get_terrain_group(task):

    """
    Group tasks for balanced terrain sampling.

    xy  : xy1~xy6
    z   : z1~z4
    xyz : xyz tasks
    other: walk/stair/no terrain constraint
    """

    task = str(task).lower()

    if "xyz" in task:
        return "xyz"

    if "xy" in task:
        return "xy"

    if "_z" in task or task.startswith("z"):
        return "z"

    return "other"




def build_terrain_sampler(dataset):

    """
    Terrain balanced sampler.

    Important:
    Do not directly read dataset.data because the stored format
    may not contain the final task field.

    Read samples through dataset.__getitem__ so that the same
    information used during training is used for sampling.
    """

    groups = []

    print("Collecting terrain groups for sampler...")

    for idx in range(len(dataset)):

        sample = dataset[idx]

        task = ""

        if isinstance(sample, dict):

            task = sample.get("task", "")

        elif hasattr(dataset, "samples"):

            item = dataset.samples[idx]

            if isinstance(item, dict):

                task = item.get("task", "")


        group = get_terrain_group(task)

        groups.append(group)


    group_count = {}

    for g in groups:

        group_count[g] = group_count.get(g, 0) + 1


    print("Terrain sampling distribution:")

    for k, v in group_count.items():

        print(f"  {k}: {v}")


    weights = []

    for g in groups:

        # smoother than inverse frequency
        weights.append(
            (1.0 / group_count[g]) ** 0.5
        )


    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler



############################################################
# Train
############################################################


def train(
        args,
        cfg
):


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(
        "Device:",
        device
    )





    ########################################################
    # GPT4TS batch adjustment
    ########################################################


    if args.model.upper() == "GPT4TS":


        if "GPT4TS" in cfg:


            if "batch_size" in cfg["GPT4TS"]:


                args.batch_size = cfg["GPT4TS"]["batch_size"]






    ########################################################
    # Dataset
    ########################################################


    train_dataset = TrigaitTerrainDataset(

        args.train

    )


    val_dataset = TrigaitTerrainDataset(

        args.val

    )





    terrain_sampler = build_terrain_sampler(
        train_dataset
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        sampler=terrain_sampler,

        num_workers=0

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0

    )








    ########################################################
    # Model
    ########################################################


    model = build_model(

        args.model,

        cfg

    ).to(device)







    ########################################################
    # Optimizer
    ########################################################


    optimizer = torch.optim.AdamW(


        model.parameters(),


        lr=cfg["learning_rate"],


        weight_decay=cfg["weight_decay"]

    )







    ########################################################
    # Output
    ########################################################


    # Experiment structure:
    # checkpoints/MODEL/SPLIT/
    # results/MODEL/SPLIT/
    # logs/MODEL/SPLIT/

    exp_name = args.name.strip("/")

    checkpoint_dir = os.path.join(

        CURRENT_DIR,

        cfg["output"]["checkpoints_dir"],

        args.model,

        exp_name

    )




    result_dir = os.path.join(

        CURRENT_DIR,

        cfg["output"]["results_dir"],

        args.model,

        exp_name

    )




    log_dir = os.path.join(

        CURRENT_DIR,

        "logs",

        args.model,

        exp_name

    )







    os.makedirs(

        checkpoint_dir,

        exist_ok=True

    )



    os.makedirs(

        result_dir,

        exist_ok=True

    )



    os.makedirs(

        log_dir,

        exist_ok=True

    )







    ########################################################
    # Save config
    ########################################################


    shutil.copy(

        args.config,

        os.path.join(

            checkpoint_dir,

            "config.yaml"

        )

    )






    best_loss = float("inf")



    train_logs = []

    val_logs = []






    ########################################################
    # Epoch
    ########################################################


    for epoch in range(

        1,

        args.epochs + 1

    ):



        print("")

        print("=" * 60)

        print(

            f"Epoch {epoch}/{args.epochs}"

        )

        print("=" * 60)






        ####################################################
        # Train
        ####################################################


        model.train()



        train_total = 0.0


        terrain_total = 0.0






        for batch in train_loader:



            imu = batch["imu"].to(device)


            terrain = batch["terrain"].to(device)

            terrain_mask = batch["terrain_mask"].to(device)


            output = model_forward(

                model,

                imu

            )



            loss_terrain = terrain_loss(

                output["terrain"],

                terrain,

                terrain_mask

            )






            # Multi-task balance
            # Increase terrain contribution to avoid xy-dominant learning
            loss = loss_terrain






            optimizer.zero_grad()



            loss.backward()




            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0

            )



            optimizer.step()






            train_total += loss.item()


            terrain_total += loss_terrain.item()








        train_loss = train_total / len(train_loader)


        train_terrain = terrain_total / len(train_loader)







        ####################################################
        # Validation
        ####################################################


        model.eval()



        val_total = 0.0


        val_terrain = 0.0






        with torch.no_grad():



            for batch in val_loader:



                imu = batch["imu"].to(device)


                terrain = batch["terrain"].to(device)

                terrain_mask = batch["terrain_mask"].to(device)




                output = model_forward(

                    model,

                    imu

                )


                loss_terrain = terrain_loss(

                    output["terrain"],

                    terrain,

                    terrain_mask

                )


                loss = loss_terrain





                val_total += loss.item()


                val_terrain += loss_terrain.item()







        val_loss = val_total / len(val_loader)


        val_terrain /= len(val_loader)






        print(

            f"Train Loss: {train_loss:.6f}"

        )


        print(

            f"Val Loss: {val_loss:.6f}"

        )






        ####################################################
        # Logs
        ####################################################


        train_logs.append({

            "epoch": epoch,

            "loss": train_loss,

            "terrain_loss": train_terrain,


            "lr":
            optimizer.param_groups[0]["lr"]

        })




        val_logs.append({

            "epoch": epoch,

            "loss": val_loss,

            "terrain_loss": val_terrain,


        })







        ####################################################
        # Save best checkpoint
        ####################################################


        if val_loss < best_loss:



            best_loss = val_loss



            save_path = os.path.join(

                checkpoint_dir,

                "best_model.pth"

            )



            torch.save(

                {

                    "epoch": epoch,

                    "model_state_dict":
                    model.state_dict(),

                    "optimizer_state_dict":
                    optimizer.state_dict(),

                    "best_loss":
                    best_loss

                },

                save_path

            )



            print(

                "Saved:",

                save_path

            )


        ####################################################
        # End epoch
        ####################################################



    ########################################################
    # Save logs
    ########################################################


    pd.DataFrame(

        train_logs

    ).to_csv(

        os.path.join(

            log_dir,

            "train_log.csv"

        ),

        index=False

    )




    pd.DataFrame(

        val_logs

    ).to_csv(

        os.path.join(

            log_dir,

            "val_log.csv"

        ),

        index=False

    )






    ########################################################
    # Save results
    ########################################################


    pd.DataFrame(

        train_logs

    ).to_csv(

        os.path.join(

            result_dir,

            "train_loss.csv"

        ),

        index=False

    )





    pd.DataFrame(

        val_logs

    ).to_csv(

        os.path.join(

            result_dir,

            "val_loss.csv"

        ),

        index=False

    )




    print("")

    print("="*60)

    print("Training finished")

    print("="*60)









############################################################
# Main
############################################################


def main():


    parser = argparse.ArgumentParser(


        description=

        "Trigait Terrain Regression Training"

    )





    parser.add_argument(

        "--model",

        type=str,

        required=True

    )





    parser.add_argument(

        "--config",

        type=str,

        required=True

    )





    parser.add_argument(

        "--train",

        type=str,

        required=True

    )





    parser.add_argument(

        "--val",

        type=str,

        required=True

    )





    parser.add_argument(

        "--epochs",

        type=int,

        default=None

    )





    parser.add_argument(

        "--batch_size",

        type=int,

        default=None

    )





    parser.add_argument(

        "--name",

        type=str,

        default="default"

    )






    args = parser.parse_args()






    ########################################################
    # Load config
    ########################################################


    cfg = load_config(

        args.config

    )







    ########################################################
    # Config override
    ########################################################


    if args.epochs is None:


        args.epochs = cfg["epochs"]






    if args.batch_size is None:


        args.batch_size = cfg["batch_size"]







    ########################################################
    # Seed
    ########################################################


    set_seed(

        cfg["seed"]

    )







    print("")

    print("="*60)

    print("Trigait Multi-task Training")

    print("="*60)


    print(

        "Model:",

        args.model

    )


    print(

        "Epochs:",

        args.epochs

    )


    print(

        "Batch size:",

        args.batch_size

    )


    print(

        "Config:",

        args.config

    )


    print("="*60)







    ########################################################
    # Start
    ########################################################


    train(

        args,

        cfg

    )







if __name__ == "__main__":


    main()
