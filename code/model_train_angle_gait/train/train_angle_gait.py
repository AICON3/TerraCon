
############################################################
# Trigait Angle + Gait Multi-task Training Framework
#
# config.yaml driven version
#
# Input:
#   IMU [B,200,42]
#
# Output:
#   angle   [B,30,8]
#   gait    [B,30,2]
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


from torch.utils.data import DataLoader



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



from dataset_angle_gait import TrigaitMultiTaskDataset





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



    # 修复：
    # 防止 Persistence 等模型没有独立参数时报错

    model_cfg = cfg.get(
        name.upper(),
        {}
    )




    ########################################################
    # TCN
    ########################################################


    if name == "tcn":


        from tcn_multitask_angle_gait import TCNMultiTask



        return TCNMultiTask(


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


        from lstm_multitask_angle_gait import LSTMMultiTask



        return LSTMMultiTask(


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


        from transformer_multitask_angle_gait import TransformerMultiTask



        return TransformerMultiTask(


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


        from gpt4ts_multitask_angle_gait import GPT4TSMultiTask



        return GPT4TSMultiTask(


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


        from timesnet_multitask_angle_gait import TimesNetMultiTask



        return TimesNetMultiTask(


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


        from patchtst_multitask_angle_gait import PatchTSTMultiTask



        return PatchTSTMultiTask(


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


        from timemixer_multitask_angle_gait import TimeMixerMultiTask



        return TimeMixerMultiTask(


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


        from mamba_multitask_angle_gait import MambaMultiTask



        return MambaMultiTask(


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

        raise ValueError(
            "Persistence is a baseline model and should not be trained."
        )






    else:


        raise ValueError(

            f"Unknown model: {name}"

        )
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

def model_forward(
        model,
        imu
):

    return model(imu)








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


    train_dataset = TrigaitMultiTaskDataset(

        args.train

    )


    val_dataset = TrigaitMultiTaskDataset(

        args.val

    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

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


    checkpoint_dir = os.path.join(

        CURRENT_DIR,

        cfg["output"]["checkpoints_dir"],

        args.model,

        args.name

    )




    result_dir = os.path.join(

        CURRENT_DIR,

        cfg["output"]["results_dir"],

        args.model,

        args.name

    )




    log_dir = os.path.join(

        CURRENT_DIR,

        "logs",

        args.model,

        args.name

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

    best_epoch = 0

    no_improve = 0

    patience = 10

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )

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


        angle_total = 0.0


        gait_total = 0.0






        for batch in train_loader:



            imu = batch["imu"].to(device)


            angle = batch["angle"].to(device)


            gait = batch["gait"].to(device)


            output = model_forward(

                model,

                imu

            )







            loss_angle = F.mse_loss(

                output["angle"],

                angle

            )






            loss_gait = F.cross_entropy(

                output["gait"].reshape(-1, 2),

                torch.argmax(

                    gait,

                    dim=-1

                ).reshape(-1)

            )
            # Multi-task balance
            # Angle is main task, gait is auxiliary task
            loss = (
                1.0 * loss_angle
                +
                0.2 * loss_gait
            )







            optimizer.zero_grad()



            loss.backward()




            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                1.0

            )



            optimizer.step()






            train_total += loss.item()


            angle_total += loss_angle.item()


            gait_total += loss_gait.item()







        train_loss = train_total / len(train_loader)


        train_angle = angle_total / len(train_loader)


        train_gait = gait_total / len(train_loader)






        ####################################################
        # Validation
        ####################################################


        model.eval()



        val_total = 0.0


        val_angle = 0.0


        val_gait = 0.0






        with torch.no_grad():



            for batch in val_loader:



                imu = batch["imu"].to(device)


                angle = batch["angle"].to(device)


                gait = batch["gait"].to(device)





                output = model_forward(

                    model,

                    imu

                )





                loss_angle = F.mse_loss(

                    output["angle"],

                    angle

                )





                loss_gait = F.cross_entropy(

                    output["gait"].reshape(-1, 2),

                    torch.argmax(

                        gait,

                        dim=-1

                    ).reshape(-1)

                )
                # Multi-task balance
                # Angle is main task, gait is auxiliary task

                loss = (
                    1.0 * loss_angle
                    +
                    0.2 * loss_gait
                )

                val_total += loss.item()

                val_angle += loss_angle.item()

                val_gait += loss_gait.item()





        val_loss = val_total / len(val_loader)


        val_angle /= len(val_loader)


        val_gait /= len(val_loader)






        print(

            f"Train Loss: {train_loss:.6f}"

        )


        print(
            f"Val Loss: {val_loss:.6f}"
        )

        print(
            f"Angle Loss: {val_angle:.6f} | "
            f"Gait Loss: {val_gait:.6f}"
        )






        scheduler.step(val_loss)

        ####################################################
        # Logs
        ####################################################


        train_logs.append({

            "epoch": epoch,

            "loss": train_loss,

            "angle_loss": train_angle,

            "gait_loss": train_gait,

            "lr":
            optimizer.param_groups[0]["lr"]

        })




        val_logs.append({

            "epoch": epoch,

            "loss": val_loss,

            "angle_loss": val_angle,

            "gait_loss": val_gait

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

            best_epoch = epoch
            no_improve = 0

        else:

            no_improve += 1

            if no_improve >= patience:

                print(
                    f"Early stopping. Best epoch: {best_epoch}"
                )

                break


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

        "Trigait Multi-task Unified Training"

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
        default=None,
        help="Checkpoint/run name. If omitted, it is inferred from --split."
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=[
            "subject_split",
            "scene_split_1",
            "scene_split_5",
            "subject_scene_split",
            "subject_5fold_split"
        ],
        help="Dataset split name; used to construct the standard experiment name."
    )






    args = parser.parse_args()

    # Resolve relative paths robustly whether the script is launched from
    # the project root or from the train/ directory.
    def resolve_input_path(path):
        if os.path.isabs(path):
            return path
        cwd_candidate = os.path.abspath(path)
        project_candidate = os.path.abspath(os.path.join(PROJECT_DIR, path))
        if os.path.isfile(cwd_candidate):
            return cwd_candidate
        return project_candidate

    args.config = resolve_input_path(args.config)
    args.train = resolve_input_path(args.train)
    args.val = resolve_input_path(args.val)

    if not os.path.isfile(args.config):
        raise FileNotFoundError("Config file not found: {}".format(args.config))
    if not os.path.isfile(args.train):
        raise FileNotFoundError("Training split file not found: {}".format(args.train))
    if not os.path.isfile(args.val):
        raise FileNotFoundError("Validation split file not found: {}".format(args.val))

    if args.name is None:
        if args.split:
            args.name = "{}_angle_gait_{}".format(args.split, args.model)
        else:
            args.name = "default"

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

