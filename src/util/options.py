"""
All command line options for training and testing
"""

import argparse


class TrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # High level training configurations
        train_config = self.parser.add_argument_group("train_config")
        train_config.add_argument("--continue_train", type=str)
        train_config.add_argument("--verbose", "-v", action="count", default=0)

        # Trainer hyperparams
        trainer_config = self.parser.add_argument_group("trainer_config")
        trainer_config.add_argument("--exp_name", type=str, default="basic")
        trainer_config.add_argument("--ema_decay", type=float, default=0.995)
        trainer_config.add_argument("--batch_size", type=int, default=64)
        trainer_config.add_argument("--lr", type=float, default=1e-4)
        trainer_config.add_argument("--num_steps", type=int, default=250_000)
        trainer_config.add_argument("--gradient_accumulate_every", type=int, default=1)
        trainer_config.add_argument("--curriculum_step", type=int, default=100_000)  # Default no curriculum
        trainer_config.add_argument("--lr_schedule", type=bool, default=False)
        trainer_config.add_argument("--amp", type=bool, default=False)
        trainer_config.add_argument("--ema_update_every", type=int, default=10)
        trainer_config.add_argument(
            "--adam_betas", nargs=2, type=float, default=(0.9, 0.99)
        )
        trainer_config.add_argument("--fp16", type=bool, default=False)
        trainer_config.add_argument("--split_batches", type=bool, default=True)
        trainer_config.add_argument("--save_every", type=int, default=2000)
        trainer_config.add_argument(
            "-p",
            "--data_path",
            type=str,
            default="../data/processed/cmu_full_500.npy",
        )
        trainer_config.add_argument(
            "-c", "--checkpoints_folder", type=str, default="../exps/"
        )

        # Diffusion hyperparams
        diffusion_config = self.parser.add_argument_group("diffusion_config")
        diffusion_config.add_argument("--T", type=int, default=500)
        diffusion_config.add_argument("--loss_type", type=str, default="l1")
        diffusion_config.add_argument("--objective", type=str, default="pred_x0")
        diffusion_config.add_argument("--beta_schedule", type=str, default="cosine")
        diffusion_config.add_argument("--p2_loss_weight_gamma", type=float, default=0.0)
        diffusion_config.add_argument("--p2_loss_weight_k", type=float, default=1)
        diffusion_config.add_argument("--t_variation", type=float, default=0.6)
        diffusion_config.add_argument("--fk_loss_lambda", type=float, default=0.2)
        diffusion_config.add_argument("--detach_fc", type=bool, default=True)
        diffusion_config.add_argument("--pos_loss", type=bool, default=True)
        diffusion_config.add_argument("--velo_loss", type=bool, default=True)

        # Unet hyperparams
        unet_config = self.parser.add_argument_group("unet_config")
        unet_config.add_argument("--channels", type=int, default=283)
        unet_config.add_argument("--dim", type=int, default=256)
        unet_config.add_argument("--dim_mults", nargs="+", type=int, default=(2, 4))
        unet_config.add_argument("--resnet_block_groups", type=int, default=8)
        unet_config.add_argument("--norm", type=str, default="group")
        unet_config.add_argument("--kernel", type=int, default=3)
        unet_config.add_argument("--stride", type=int, default=2)
        unet_config.add_argument("--padding", type=int, default=1)


class TestOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        test_config = self.parser.add_argument_group("test_config")
        test_config.add_argument("--verbose", "-v", action="count", default=0)
        test_config.add_argument("--exp_names", nargs="+", type=str)
        test_config.add_argument("--milestones", "-n", nargs="+", type=int, default=0)
        test_config.add_argument("--mode", "-m", type=str, default="unconditional")
        test_config.add_argument("--n_samples", "-s", type=int, default=4)
        test_config.add_argument("--sample_len", "-l", type=int, default=2000)
