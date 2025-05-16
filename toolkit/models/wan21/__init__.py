from .wan21 import Wan21
try:
    from .wan21_i2v import Wan21I2V
except ImportError as e:
    print(f"Warning: Could not import Wan21I2V: {e}")
    pass