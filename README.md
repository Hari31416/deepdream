# DeapDream

DeapDream is a project that uses the DeepDream algorithm to generate images. The DeepDream algorithm is a computer vision algorithm that uses a convolutional neural network to enhance patterns in images. The algorithm works by maximizing the activation of a specific layer in the neural network. This results in the enhancement of patterns and features in the image, creating a dream-like effect.

## How to Use

The main class `DeepDream` implements all the required logics to create a DeepDream image. Two important parameters are required to create a DeepDream image:

- `base_model`: The pre-trained model to use for the DeepDream algorithm. - `block_names_to_select`: The blocks that will be used to generate the DeepDream image.

After creating an instance of the `DeepDream` class, you can call the `dream` method to generate the DeepDream image. The method takes the following parameters:

```text
image: PIL.Image
    The image on which to perform DeepDream.
iterations: int, optional
    The number of iterations to perform, by default 20
lr: float, optional
    The learning rate to use for gradient ascent, by default 0.05
octave_scale: float, optional
    The scale to use for resizing the image in each octave, by default 1.3
octaves: List[int], optional
    A list of octaves to use for DeepDream, by default [0], which means only one octave is used
loss_type: str, optional
    Loss function to use. Must be either 'mean' or 'norm'. By Default 'mean'
plot_image_interval: Union[str, int], optional
    The interval with which to plot the dreamed image. If None, no image is plotted. If string, must be equal to 'final_image' which plots the final image.
```

Here is an example of how to use the `DeepDream` class to generate a DeepDream image:

```python
from torchvision.models import inception_v3 Inception_V3_Weights
from deepdream.deepdream import DeepDream
import PIL.Image

base_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
block_names_to_select = ["Mixed_5b", "Mixed_5c", "Mixed_5d"]
dd = DeepDream(base_model, block_names_to_select)
sample_image = PIL.Image.open("sample_images/sky1024px.jpg")
dreamed = dd.dream(
    sample_image,
    iterations=10,
    lr=0.1,
    octaves=[-2, -1, 0, 1, 2],
    loss_type="mean",
    plot_image_interval=2,
)
```

The `deepdream` module implements some helper methods that can be used to get a look at the architecture of the model and thus to decide which blocks to select for the DeepDream algorithm. The method `get_model_tree` returns a dictionary of the model tree of the model. This tree can be used with the `print_model_tree` prints the model tree of the model. `create_all_possible_submodule_keys` method can be used to get all the possible submodule keys of the model. Any of these keys can be used to select the blocks for the DeepDream algorithm.
