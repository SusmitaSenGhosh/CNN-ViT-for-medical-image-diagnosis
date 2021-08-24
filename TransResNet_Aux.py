import warnings
import tensorflow as tf
import tensorflow_addons as tfa
import layers, utils

CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": [64,64,256,512,512],
    "num_heads": 8,
    "num_layers": 4,
    "hidden_size": 512,
}


def transformer_outerblock(my_input,my_num_layers,my_num_heads,my_mlp_dim,my_dropout,my_pool_size):
    my_output = tf.keras.layers.AveragePooling2D(pool_size=(1,my_pool_size),data_format='channels_first')(my_input)
    my_output = tf.keras.layers.Reshape((my_output.shape[1] * my_output.shape[2], my_output.shape[3]))(my_output)
    my_output = layers.ClassToken()(my_output)
    my_output= layers.AddPositionEmbs()(my_output)
    for n in range(my_num_layers):
        my_output, _ = layers.TransformerBlock(
            num_heads=my_num_heads,
            mlp_dim=my_mlp_dim,
            dropout=my_dropout
        )(my_output)
    my_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6
    )(my_output)
    my_output = tf.keras.layers.Lambda(lambda v: v[:, 0])(my_output)
    return my_output

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
    x1 = base_model.get_layer("pool1_pool").output
    x2 = base_model.get_layer("conv2_block3_out").output
    x3 = base_model.get_layer("conv3_block4_out").output
    x4 = base_model.get_layer("conv4_block6_out").output
    x5 = base_model.get_layer("conv5_block2_out").output

    # print(x1.shape)
    # print(x2.shape)
    print(x3.shape)
    # print(x4.shape)
    # print(x5.shape)
    # y1 = transformer_outerblock(x1, num_layers, num_heads, mlp_dim[0], dropout,1)
    # y1 = tf.keras.layers.BatchNormalization()(y1)
    # y1 = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(y1)
    # y1 = tf.keras.layers.BatchNormalization()(y1)
    # y1 = tf.keras.layers.Dense(3, 'softmax')(y1)

    # y2 = transformer_outerblock(x2, num_layers, num_heads, mlp_dim[1], dropout,1)
    # y2 = tf.keras.layers.BatchNormalization()(y2)
    # y2 = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(y2)
    # y2 = tf.keras.layers.BatchNormalization()(y2)
    # y2 = tf.keras.layers.Dense(8, 'softmax')(y2)
    
    y3 = tf.keras.layers.AveragePooling2D(pool_size = (2,2))(x3)
    print(y3.shape)
    y3 = transformer_outerblock(y3, 1, num_heads, mlp_dim[2], dropout,1)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(y3)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(y3)
    y3 = tf.keras.layers.BatchNormalization()(y3)
    y3 = tf.keras.layers.Dense(classes, 'softmax',name = 'output0')(y3)
    
    y4 = transformer_outerblock(x4, 1, num_heads, mlp_dim[3], dropout,2)
    y4 = tf.keras.layers.BatchNormalization()(y4)
    y4 = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(y4)
    y4 = tf.keras.layers.BatchNormalization()(y4)
    y4 = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(y4)
    y4 = tf.keras.layers.BatchNormalization()(y4)
    y4 = tf.keras.layers.Dense(classes, 'softmax',name = 'output1')(y4)
    
    y5 = transformer_outerblock(x5, 2, num_heads, mlp_dim[4], dropout,4)
    y5 = tf.keras.layers.BatchNormalization()(y5)
    y5 = tf.keras.layers.Dense(128, activation = tfa.activations.gelu)(y5)
    y5 = tf.keras.layers.BatchNormalization()(y5)
    y5 = tf.keras.layers.Dense(32, activation = tfa.activations.gelu)(y5)
    y5 = tf.keras.layers.BatchNormalization()(y5)
    y5 = tf.keras.layers.Dense(classes, 'softmax',name = 'output2')(y5)
    
    return tf.keras.models.Model(inputs=base_model.inputs, outputs=[y3,y4,y5], name=name)


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
        patch_size=16,
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