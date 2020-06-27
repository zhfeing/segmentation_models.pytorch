import torch

import segmentation_models_pytorch as smp


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")


x = torch.ones(2, 3, 224, 224).to(device)
for model_name, config in smp.encoders.encoders.items():
    pretrained_settings = config["pretrained_settings"]
    for dataset_name in pretrained_settings.keys():
        print("generating encoder: {} in dataset: {}".format(model_name, dataset_name))
        model = smp.Unet(
            encoder_name=model_name,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
            classes=10,
            encoder_weights=dataset_name
        ).to(device)

        y = model(x)
        print(y.shape)
