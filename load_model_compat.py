# load_model_compat.py
import zipfile
import json
import io
import tensorflow as tf

def load_model_compatible(filepath):
    """Load a .keras model file, stripping unsupported quantization_config fields."""

    # Read the .keras zip and patch the config
    with zipfile.ZipFile(filepath, 'r') as zf:
        file_list = zf.namelist()
        files = {name: zf.read(name) for name in file_list}

    # Parse and clean the config
    config = json.loads(files['config.json'])
    config = _remove_quantization_config(config)
    files['config.json'] = json.dumps(config).encode('utf-8')

    # Write patched zip to memory buffer
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf_out:
        for name, data in files.items():
            zf_out.writestr(name, data)
    buffer.seek(0)

    # Save patched file temporarily and load
    patched_path = filepath.replace('.keras', '_patched.keras')
    with open(patched_path, 'wb') as f:
        f.write(buffer.read())

    return tf.keras.models.load_model(patched_path, compile=False)


def _remove_quantization_config(obj):
    """Recursively remove quantization_config keys from config dicts."""
    if isinstance(obj, dict):
        return {
            k: _remove_quantization_config(v)
            for k, v in obj.items()
            if k != 'quantization_config'
        }
    elif isinstance(obj, list):
        return [_remove_quantization_config(item) for item in obj]
    return obj