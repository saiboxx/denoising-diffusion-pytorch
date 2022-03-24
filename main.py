import torch

from ddpm import Unet, GaussianDiffusion, Trainer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main() -> None:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=2
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=128,
        timesteps=1000,  # number of steps
        loss_type='l1',  # L1 or L2
        device=DEVICE,
    )

    trainer = Trainer(
        diffusion,
        '/projects/core-rad/data/ILSVRC2012_img_train',
        train_batch_size=64,
        train_lr=2e-5,
        train_num_steps=1000000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        fp16=True, # turn on mixed precision training
        save_and_sample_every=1000,
        num_workers=64,
        results_folder='results/v2'
    )

    trainer.train()


if __name__ == '__main__':
    main()
