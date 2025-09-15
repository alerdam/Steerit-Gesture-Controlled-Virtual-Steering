import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

def inspect_model(model: Model, show_trainable_only=False):
    print("="*60)
    print("🧠 MODEL SUMMARY")
    print("="*60)
    model.summary()
    
    print("\n📐 INPUT SHAPE")
    print("-"*60)
    for i, input_tensor in enumerate(model.inputs):
        print(f"Input {i+1}: {input_tensor.shape}")

    print("\n📤 OUTPUT SHAPE")
    print("-"*60)
    for i, output_tensor in enumerate(model.outputs):
        print(f"Output {i+1}: {output_tensor.shape}")

    print("\n🔍 LAYER DETAILS")
    print("-"*60)
    for i, layer in enumerate(model.layers):
        if show_trainable_only and not layer.trainable:
            continue
        print(f"[{i:02d}] {layer.name:<25} | {layer.__class__.__name__:<20} | Trainable: {layer.trainable}")
        try:
            print(f"     └─ Input shape: {layer.input_shape}")
            print(f"     └─ Output shape: {layer.output_shape}")
        except AttributeError:
            pass
        print(f"     └─ Params: {layer.count_params()}")
        print("-"*60)

    total_params = model.count_params()
    trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
    non_trainable_params = total_params - trainable_params

    print("\n📊 PARAMETER COUNT")
    print("-"*60)
    print(f"Total params:        {total_params:,}")
    print(f"Trainable params:    {trainable_params:,}")
    print(f"Non-trainable params:{non_trainable_params:,}")
    print("="*60)

# Example usage:
modelpath = f"M_timeaware_rnn.keras"
model = keras.models.load_model(modelpath)
inspect_model(model)