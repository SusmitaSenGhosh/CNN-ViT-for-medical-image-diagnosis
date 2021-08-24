import warnings
import tensorflow as tf
import layers, utils

CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": 512,
    "num_heads": 8,
    "num_layers": 4,
    "hidden_size": 512,
}

def build_model(
    image_size: int,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    activation="linear",
    include_top=True,
    representation_size=None,
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
   # x = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    base_model = tf.keras.applications.ResNet50(include_top=False, input_tensor=None, input_shape=(224,224,3), pooling='avg')
    base_model.trainable = True
    x = base_model.get_layer("conv5_block2_out").output
    
    
    # y = tf.keras.layers.Conv2D(
    #     filters=hidden_size,
    #     kernel_size=patch_size,
    #     strides=patch_size,
    #     padding="valid",
    #     name="embedding",
    # )(x)
    y = tf.keras.layers.Reshape((x.shape[1] * x.shape[2], x.shape[3]))(x)
    y = tf.keras.layers.AveragePooling1D(pool_size=4,data_format='channels_first')(y)
    y = layers.ClassToken(name="class_token")(y)
    y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh"
        )(y)
    if include_top:
        y = tf.keras.layers.Dense(classes, name="head", activation=activation)(y)
    return tf.keras.models.Model(inputs=base_model.inputs, outputs=y, name=name)


def my_model(
    image_size: int = 224,
    classes=1000,
    activation="linear",
    include_top=False,
    pretrained=False,
    pretrained_top=False,
    weights="imagenet21k+imagenet2012",
):

    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        patch_size=7,
        image_size=image_size,
        classes=classes,
        activation=activation,
        include_top=include_top,
        representation_size=768 if weights == "imagenet21k" else None,
    )
    # if pretrained:
    #     load_pretrained(
    #         size="B_16", weights=weights, model=model, pretrained_top=pretrained_top
    #     )
    return model