# FILE: backend/vit_model.py
# PURPOSE: Final Hybrid ViT-UNet structure, stable for training.

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np

# --- 0. Custom Loss Function Fix (Required for loading) ---
def CustomBinaryCrossentropy(y_true, y_pred):
    """
    Forces both predicted and true values to flatten into single vectors 
    before loss calculation to bypass Keras's internal shape conflict bug.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true_flat = tf.reshape(y_true, shape=[-1])
    y_pred_flat = tf.reshape(y_pred, shape=[-1])
    return tf.keras.losses.binary_crossentropy(y_true_flat, y_pred_flat)


# --- 1. ViT Helper Components: Patching and Transformer Block ---

# Removed unused image_size and num_channels arguments from the definition
def PatchEmbed(patch_size, embed_dim):
    """
    Keras Sequential model for patch embedding, replacing Conv2D and Reshape.
    """
    return tf.keras.Sequential([
        # The Conv2D layer handles both patch extraction and embedding (dimensionality C -> embed_dim)
        layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid", name="conv_patch_embed"),
        # Reshape to (num_patches, embed_dim) for the Transformer blocks
        layers.Reshape((-1, embed_dim), name="reshape_patch_embed") 
    ], name="patch_embed")

def TransformerBlock(embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1, name="transformer_block"):
    inputs = layers.Input(shape=(None, embed_dim))
    
    norm1 = layers.LayerNormalization(epsilon=1e-6)(inputs)
    # The key_dim is typically set to embed_dim/num_heads for standard attention, 
    # but setting key_dim=embed_dim is also common in Keras implementations.
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=dropout_rate)(norm1, norm1)
    
    x1 = layers.Add()([inputs, attn_output])
    
    norm2 = layers.LayerNormalization(epsilon=1e-6)(x1)
    ffn = layers.Dense(feed_forward_dim, activation="gelu")(norm2)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(embed_dim)(ffn)
    
    outputs = layers.Add()([x1, ffn])
    
    return Model(inputs=inputs, outputs=outputs, name=name)

# --- 2. Hybrid ViT-UNet Model Definition ---

def build_vit_segmentation_model(input_shape=(256, 256, 1), patch_size=16, num_layers=4, num_heads=4, embed_dim=64, feed_forward_dim=128, decoder_filters=(64, 32, 16, 8)):
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    H, W, C = input_shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    
    # --- ENCODER (ViT Backbone) ---
    # Call PatchEmbed with only patch_size and embed_dim
    patch_embed_layer = PatchEmbed(patch_size, embed_dim)
    x = patch_embed_layer(inputs)
    
    # Positional Embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embeddings = layers.Embedding(input_dim=num_patches, output_dim=embed_dim, name="pos_embedding")(positions)
    pos_embeddings = tf.expand_dims(pos_embeddings, axis=0)
    x = layers.Add(name="add_pos_embeddings")([x, pos_embeddings])

    # Transformer Blocks
    for i in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, feed_forward_dim, name=f"transformer_block_{i}")(x)
    
    # --- DECODER (Tracing Head - UNet-like structure) ---
    # Reshape the sequence back to a grid structure
    x = layers.Reshape((num_patches_h, num_patches_w, embed_dim), name="reshape_decoder_start")(x)
    
    y = x
    # Upsampling using Conv2DTranspose (Decodes the latent space)
    for idx, f in enumerate(decoder_filters):
        y = layers.Conv2DTranspose(filters=f, kernel_size=3, strides=2, padding="same", activation="relu", name=f"decoder_conv2dtranspose_{idx}")(y)
        y = layers.BatchNormalization(name=f"decoder_bn_{idx}")(y)
    
    # Final 1x1 convolution for single channel output (edge mask)
    outputs = layers.Conv2D(filters=1, kernel_size=1, activation="sigmoid", padding="same", name="final_mask_conv")(y)
    
    model = Model(inputs=inputs, outputs=outputs, name="Hybrid_ViT_Edge_Detector")
    model.compile(optimizer='adam', loss=CustomBinaryCrossentropy, metrics=['accuracy'])
    return model