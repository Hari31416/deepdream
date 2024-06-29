from deepdream.deepdream import (
    DeepDream,
    print_model_tree,
    create_all_possible_submodule_keys,
    get_model_tree,
    get_transforms,
    deprocess,
)

import gradio as gr
import PIL.Image

available_models = [
    "alexnet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
    "googlenet",
    "inception_v3",
    "mobilenet_v2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "vgg16_bn",
    "vgg19_bn",
]


def load_model(model_name):
    from torchvision.models import get_model, get_model_weights

    if model_name not in available_models:
        raise ValueError(
            f"Model {model_name} not available. Available models are: {available_models}"
        )

    model = get_model(model_name, weights=get_model_weights(model_name).IMAGENET1K_V1)
    return model


def dream(
    submodules_selected,
    image,
    iterations,
    lr,
    octave_scale,
    octaves,
    loss_type,
):
    import torch.optim as optim

    # select_model = gr.Dropdown(
    #     choices=available_models,
    #     label="Select model",
    # )
    # # load model
    model = load_model("inception_v3")

    dd = DeepDream(
        base_model=model,
        block_names_to_select=submodules_selected,
    )

    original_shape = image.size
    # PIL has (w, h) format
    original_shape = tuple(reversed(original_shape))
    transform = get_transforms(*original_shape)
    new_image = transform(image).to(dd.device)
    for i in octaves:
        new_size = tuple([int(dim * (octave_scale**i)) for dim in original_shape])
        new_image = dd.resize_tf_image(
            new_image.detach(), new_size, get_transforms(*new_size)
        )
        model = dd.dream_model
        new_image = new_image.unsqueeze(0).clone()  # add batch dimension
        new_image = new_image.to(dd.device)
        new_image.requires_grad = True
        optimizer = optim.Adam([new_image], lr=lr)
        if loss_type == "mean":
            dd.logger.debug("Using mean of activations for loss")
            loss_fn = dd.activation_loss_mean
        elif loss_type == "norm":
            dd.logger.debug("Using norm of activations for loss")
            loss_fn = dd.activation_loss_norm
        else:
            m = f"`loss_type` must either be mean or norm, found: {loss_type}"
            dd.logger.error(m)
            raise ValueError(m)

        for i in range(iterations):
            optimizer.zero_grad()
            activations = model(new_image)
            loss = loss_fn(activations)
            loss.backward()
            dd.normalize_grad(new_image)
            dd.logger.info(f"Iteration {i+1}/{iterations}, Loss: {loss.item()}")
            optimizer.step()
            # clip the new_image
            new_image.data = new_image.data.clamp_(-1, 1)
            yield deprocess(new_image), -loss.item(), i


def update_second(first_val):
    model = load_model(first_val)
    tree = get_model_tree(model)
    all_keys = create_all_possible_submodule_keys(tree)
    d2 = gr.Dropdown(choices=all_keys)
    return d2


def main():
    import gc

    model = load_model("inception_v3")
    tree = get_model_tree(model)
    all_keys = create_all_possible_submodule_keys(tree)

    submodules_selected = gr.Dropdown(
        choices=all_keys,
        multiselect=True,
        label="Select submodules",
    )
    del model
    gc.collect()

    # upload image
    uploaded_file = gr.Image(label="Upload image", type="pil")
    iterations = gr.Slider(minimum=1, maximum=100, value=20, label="Iterations")
    lr = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, label="Learning rate")
    octave_scale = gr.Slider(minimum=1.1, maximum=1.9, value=1.3, label="Octave scale")
    octaves = gr.Dropdown(
        choices=[-4, -3, -2, -1, 0, 1, 2, 3, 4],
        multiselect=True,
        label="Octaves",
        value=[0],
    )
    loss_type = gr.Dropdown(choices=["mean", "norm"], label="Loss type", value="mean")
    inputs = [
        submodules_selected,
        uploaded_file,
        iterations,
        lr,
        octave_scale,
        octaves,
        loss_type,
    ]
    outputs = [
        gr.Image(label="Deep dream image"),
        gr.Label(label="Loss"),
        gr.Label(label="Iteration"),
    ]

    interface = gr.Interface(
        dream,
        inputs=inputs,
        outputs=outputs,
        title="Deep Dream",
        description="Deep dream",
        examples=[
            [
                [
                    "Mixed_5d",
                    "Mixed_6a",
                    "Mixed_6e",
                ],
                PIL.Image.open("sample_images/sky1024px.jpg"),
                10,
                0.1,
                1.3,
                [0],
                "mean",
            ]
        ],
    )
    interface.queue().launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
