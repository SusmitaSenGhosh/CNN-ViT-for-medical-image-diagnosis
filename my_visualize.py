import cv2
import numpy as np
import tensorflow as tf
import layers
import matplotlib.cm as cm

def attention_map(model, image,alpha):
    """Get an attention map for an image and model using the technique
    described in Appendix D.7 in the paper (unofficial).

    Args:
        model: A ViT model
        image: An image for which we will compute the attention map.
    """
    size = model.input_shape[1]
    grid_size = 7#int(np.sqrt(model.layers[-13].output_shape[0][-2] - 1))

    # Prepare the input
    X = image#vit.preprocess_inputs(cv2.resize(image, (size, size)))[np.newaxis, :]  # type: ignore
    
    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if isinstance(l, layers.TransformerBlock)
    ]
    # print(len(outputs),X.shape)
    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    print(weights.shape)
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    reshaped = weights.reshape(num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)
    print(weights.shape)

    # )

    # # # From Appendix D.6 in the paper ...
    # # # Average the attention weights across all heads.
    reshaped = reshaped.mean(axis=1)

    # # # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # # # To account for residual connections, we add an identity matrix to the
    # # # attention matrix and re-normalize the weights.
    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[2]))
    heatmap = np.uint8(255 * mask)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    #jet_heatmap = jet_heatmap/jet_heatmap.max()

    # Superimpose the heatmap on original image
    # print(jet_heatmap.max(),jet_heatmap.min())
    image = image
    superimposed_img = jet_heatmap* alpha + np.squeeze(image)
    superimposed_img = superimposed_img/superimposed_img.max()
    return jet_heatmap,superimposed_img
