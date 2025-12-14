import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os
import logging
import json
from pathlib import Path
import socket
import time
import sys
import subprocess

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

# Disable Gradio update checks
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"
# Prevent Gradio version check
os.environ["GRADIO_SKIP_VERSION_CHECK"] = "True"


def _is_gpu_available():
    """Check if OpenCV has CUDA support enabled."""
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


GPU_ENABLED = False

# Pydantic compatibility fix
try:
    import pydantic
    import fastapi
    from fastapi import FastAPI
    
    # Check Pydantic version and add necessary compatibility
    pydantic_version = getattr(pydantic, '__version__', '0')
    # Remove version output to keep logs clean
    # print(f"Using Pydantic version {pydantic_version}")
    
    # Add patches for Pydantic v1/v2 compatibility
    if pydantic_version.startswith('2.'):
        # Create fake RootModel for compatibility
        if not hasattr(pydantic, 'RootModel'):
            class RootModelV1Compat(pydantic.BaseModel):
                root: object
                
                def __init__(self, root=None, **kwargs):
                    super().__init__(root=root, **kwargs)
                
                @property
                def model_dump(self):
                    return lambda *args, **kwargs: {'root': self.root}
                    
            setattr(pydantic, 'RootModel', RootModelV1Compat)
        
        # Patch for Starlette Request in Pydantic v2
        orig_fastapi_init = fastapi.FastAPI.__init__
        
        def patched_fastapi_init(self, *args, **kwargs):
            if 'model_config' not in kwargs:
                kwargs['model_config'] = {'arbitrary_types_allowed': True}
            elif isinstance(kwargs['model_config'], dict):
                kwargs['model_config']['arbitrary_types_allowed'] = True
            return orig_fastapi_init(self, *args, **kwargs)
        
        fastapi.FastAPI.__init__ = patched_fastapi_init
        
        # Enhanced patch for fastapi._compat, solving issues with starlette.requests.Request
        if hasattr(fastapi, '_compat'):
            if hasattr(fastapi._compat, 'get_model_fields'):
                # Original function
                orig_get_model_fields = fastapi._compat.get_model_fields
                
                # Replacement for safe handling of Request and other types
                def safe_get_model_fields(model):
                    try:
                        from starlette.requests import Request
                        # If model is Request, return empty dict
                        if model == Request or getattr(model, '__name__', '') == 'Request':
                            return {}
                        return orig_get_model_fields(model)
                    except Exception:
                        return {}
                
                # Set the safe function
                fastapi._compat.get_model_fields = safe_get_model_fields
    else:
        # Patches for Pydantic v1
        # Should already be compatible with FastAPI/Starlette
        pass
        
    # Suppress BaseSettings import error
    orig_import = __import__
    
    def custom_import(name, *args, **kwargs):
        if name == 'pydantic' and 'BaseSettings' in args[2]:
            # Show warning only once
            if not hasattr(sys, '_showed_basesettings_warning'):
                print("Info: BaseSettings has been moved to pydantic-settings package in Pydantic v2")
                setattr(sys, '_showed_basesettings_warning', True)
        return orig_import(name, *args, **kwargs)
    
    # Don't change __import__ to our custom one - this might cause other problems
    # sys.modules['builtins'].__import__ = custom_import
    
except Exception as e:
    import traceback
    print(f"Warning: failed to apply Pydantic patch: {e}")
    print(traceback.format_exc())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lutplus")

# Disable only Gradio's internal logging
logging.getLogger('gradio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)
logging.getLogger('pydantic').setLevel(logging.WARNING)

# Presets handling
PRESETS_FILE = "presets.json"

def load_presets():
    try:
        if os.path.exists(PRESETS_FILE):
            with open(PRESETS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading presets: {str(e)}")
    return {}

def save_preset(name, settings):
    presets = load_presets()
    presets[name] = settings
    try:
        with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, indent=2, ensure_ascii=False)
        return list(presets.keys())
    except Exception as e:
        print(f"Error saving preset: {str(e)}")
        return list(presets.keys())

def delete_preset(name):
    presets = load_presets()
    if name in presets:
        del presets[name]
        try:
            with open(PRESETS_FILE, 'w', encoding='utf-8') as f:
                json.dump(presets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving presets after deletion: {str(e)}")
    return list(presets.keys())

def apply_preset(preset_name):
    presets = load_presets()
    if preset_name in presets:
        settings = presets[preset_name]
        return [
            settings.get('noise_level', 0),
            settings.get('blur', 1),
            settings.get('brightness', 1.0),
            settings.get('saturation', 1.0),
            settings.get('temperature', 0),
            settings.get('lut_intensity', 0.5)
        ]
    return [0, 1, 1.0, 1.0, 0, 0.5]

def apply_color_noise(image, level):
    if level == 0:
        return image
    img = image.astype(np.float32)
    noise = np.random.normal(0, level, img.shape).astype(np.float32)
    noisy_img = cv2.add(img, noise)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_monochrome_noise(image, level):
    if level == 0:
        return image
    img = image.astype(np.float32)
    # Generate single channel noise
    noise = np.random.normal(0, level, img.shape[:2]).astype(np.float32)
    # Apply same noise to all channels
    noise_3ch = np.stack([noise] * 3, axis=-1)
    noisy_img = cv2.add(img, noise_3ch)
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_gaussian_noise(image, level):
    if level == 0:
        return image
    img = image.astype(np.float32)
    # Gaussian noise with mean=0
    noise = np.random.normal(0, level, img.shape).astype(np.float32)
    noisy_img = img + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_digital_grain(image, level):
    if level == 0:
        return image
    img = image.astype(np.float32)
    # Create grain pattern
    grain = np.random.uniform(-level, level, img.shape[:2]).astype(np.float32)
    # Apply threshold to create more distinct grain
    grain = np.where(grain > level/2, level, -level)
    # Stack for all channels but with reduced intensity in color channels
    grain_color = np.stack([grain, grain * 0.5, grain * 0.5], axis=-1)
    noisy_img = img + grain_color
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def apply_gaussian_blur(image, kernel_size):
    if image is None:
        return None
    # Ensure kernel size is odd
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if GPU_ENABLED:
        try:
            return apply_gaussian_blur_gpu(image, kernel_size)
        except Exception as exc:
            logger.warning(f"GPU blur failed, falling back to CPU: {exc}")

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_gaussian_blur_gpu(image, kernel_size):
    """Apply Gaussian blur using CUDA if available."""
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)
    gaussian = cv2.cuda.createGaussianFilter(
        gpu_image.type(),
        gpu_image.type(),
        (kernel_size, kernel_size),
        0
    )
    blurred_gpu = gaussian.apply(gpu_image)
    return blurred_gpu.download()

def adjust_brightness(image, factor):
    if image is None:
        return None
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def adjust_saturation(image, factor):
    if image is None:
        return None
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Adjust saturation
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    # Clip values
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    # Convert back to RGB
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result

def adjust_color_temperature(image, temperature):
    if image is None:
        return None
    result = image.copy()
    if temperature > 0:  # Warm
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 + temperature/100), 0, 255)  # Red
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 - temperature/200), 0, 255)  # Blue
    else:  # Cool
        temperature = abs(temperature)
        result[:, :, 2] = np.clip(result[:, :, 2] * (1 + temperature/100), 0, 255)  # Blue
        result[:, :, 0] = np.clip(result[:, :, 0] * (1 - temperature/200), 0, 255)  # Red
    return result.astype(np.uint8)

def adjust_gamma(image, gamma):
    if image is None or gamma == 1.0:
        return image
    # Ensure gamma is not zero or negative
    gamma = max(0.01, gamma)  # Limit minimum gamma to avoid division by zero
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def adjust_contrast(image, factor):
    if image is None or factor == 1.0:
        return image

    mean = np.mean(image)
    adjusted = cv2.addWeighted(image, factor, image, 0, mean * (1 - factor))
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def gpu_available():
    """Check if PyTorch with CUDA is available."""
    if not TORCH_AVAILABLE:
        return False, "PyTorch not installed"
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            device_name = "CUDA device"
        return True, device_name
    return False, "CUDA not detected"


def to_torch_image(image, device):
    tensor = torch.as_tensor(image, dtype=torch.float32, device=device)
    if tensor.ndim == 3:
        return tensor
    return tensor.view(*image.shape)


def torch_gaussian_kernel(kernel_size: int, sigma: float, device):
    coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
    grid = coords ** 2
    kernel_1d = torch.exp(-grid / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.expand(3, 1, kernel_size, kernel_size)
    return kernel_2d


def torch_gaussian_blur(image, kernel_size):
    if kernel_size <= 1:
        return image
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 6.0
    kernel = torch_gaussian_kernel(kernel_size, sigma, image.device)
    img = image.permute(2, 0, 1).unsqueeze(0)  # NCHW
    padding = kernel_size // 2
    blurred = torch.nn.functional.conv2d(img, kernel, padding=padding, groups=3)
    return blurred.squeeze(0).permute(1, 2, 0)


def torch_adjust_saturation(image, factor):
    if factor == 1.0:
        return image
    img = image / 255.0
    maxc, _ = img.max(dim=2)
    minc, _ = img.min(dim=2)
    delta = maxc - minc
    s = torch.zeros_like(maxc)
    non_zero = maxc != 0
    s[non_zero] = delta[non_zero] / maxc[non_zero]

    # Compute hue
    r, g, b = img.unbind(-1)
    h = torch.zeros_like(maxc)
    mask = delta != 0
    r_eq_max = (maxc == r) & mask
    g_eq_max = (maxc == g) & mask
    b_eq_max = (maxc == b) & mask
    h[r_eq_max] = ((g - b) / delta % 6)[r_eq_max]
    h[g_eq_max] = ((b - r) / delta + 2)[g_eq_max]
    h[b_eq_max] = ((r - g) / delta + 4)[b_eq_max]
    h = h / 6

    s = torch.clamp(s * factor, 0, 1)

    i = torch.floor(h * 6).long()
    f = h * 6 - i
    p = maxc * (1 - s)
    q = maxc * (1 - f * s)
    t = maxc * (1 - (1 - f) * s)

    i_mod = i % 6
    conditions = [
        (i_mod == 0, torch.stack([maxc, t, p], dim=-1)),
        (i_mod == 1, torch.stack([q, maxc, p], dim=-1)),
        (i_mod == 2, torch.stack([p, maxc, t], dim=-1)),
        (i_mod == 3, torch.stack([p, q, maxc], dim=-1)),
        (i_mod == 4, torch.stack([t, p, maxc], dim=-1)),
        (i_mod == 5, torch.stack([maxc, p, q], dim=-1)),
    ]

    rgb = torch.zeros_like(img)
    for condition, value in conditions:
        rgb = torch.where(condition.unsqueeze(-1), value, rgb)
    return torch.clamp(rgb * 255.0, 0, 255)


def apply_lut_torch(image, table, intensity):
    size = table.shape[0]
    img = image.float() / 255.0
    h, w, _ = img.shape
    coords = torch.clamp(img * (size - 1), 0, size - 1)
    coords_floor = torch.floor(coords).long()
    coords_ceil = torch.clamp(coords_floor + 1, max=size - 1)
    weights = coords - coords_floor.float()

    xf, yf, zf = coords_floor.unbind(-1)
    xc, yc, zc = coords_ceil.unbind(-1)
    wx, wy, wz = weights.unbind(-1)

    v000 = table[xf, yf, zf]
    v001 = table[xf, yf, zc]
    v010 = table[xf, yc, zf]
    v011 = table[xf, yc, zc]
    v100 = table[xc, yf, zf]
    v101 = table[xc, yf, zc]
    v110 = table[xc, yc, zf]
    v111 = table[xc, yc, zc]

    v00 = v000 * (1 - wx.unsqueeze(-1)) + v100 * wx.unsqueeze(-1)
    v01 = v001 * (1 - wx.unsqueeze(-1)) + v101 * wx.unsqueeze(-1)
    v10 = v010 * (1 - wx.unsqueeze(-1)) + v110 * wx.unsqueeze(-1)
    v11 = v011 * (1 - wx.unsqueeze(-1)) + v111 * wx.unsqueeze(-1)

    v0 = v00 * (1 - wy.unsqueeze(-1)) + v10 * wy.unsqueeze(-1)
    v1 = v01 * (1 - wy.unsqueeze(-1)) + v11 * wy.unsqueeze(-1)

    result = v0 * (1 - wz.unsqueeze(-1)) + v1 * wz.unsqueeze(-1)

    if intensity < 1.0:
        result = img * (1 - intensity) + result * intensity

    return torch.clamp(result * 255.0, 0, 255)


def process_image_torch(image, color_noise, mono_noise, gauss_noise, digital_grain,
                        blur_kernel, brightness, contrast, saturation, temperature,
                        gamma, lut_file, lut_intensity, device):
    result = to_torch_image(image, device)

    if color_noise > 0:
        noise = torch.normal(0, float(color_noise), size=result.shape, device=device)
        result = torch.clamp(result + noise, 0, 255)

    if mono_noise > 0:
        noise = torch.normal(0, float(mono_noise), size=result.shape[:2], device=device)
        noise = noise.unsqueeze(-1).expand_as(result)
        result = torch.clamp(result + noise, 0, 255)

    if gauss_noise > 0:
        noise = torch.normal(0, float(gauss_noise), size=result.shape, device=device)
        result = torch.clamp(result + noise, 0, 255)

    if digital_grain > 0:
        grain = torch.empty(result.shape[:2], device=device).uniform_(-digital_grain, digital_grain)
        grain = torch.where(grain > digital_grain / 2, digital_grain, -digital_grain)
        grain_color = torch.stack([grain, grain * 0.5, grain * 0.5], dim=-1)
        result = torch.clamp(result + grain_color, 0, 255)

    if blur_kernel > 1:
        result = torch_gaussian_blur(result, blur_kernel)

    if brightness != 1.0:
        result = torch.clamp(result * brightness, 0, 255)

    if contrast != 1.0:
        mean = result.mean()
        result = torch.clamp(result * contrast + mean * (1 - contrast), 0, 255)

    if saturation != 1.0:
        result = torch_adjust_saturation(result, saturation)

    if temperature != 0:
        result = result.clone()
        if temperature > 0:
            result[..., 0] = torch.clamp(result[..., 0] * (1 + temperature / 100), 0, 255)
            result[..., 2] = torch.clamp(result[..., 2] * (1 - temperature / 200), 0, 255)
        else:
            temp = abs(temperature)
            result[..., 2] = torch.clamp(result[..., 2] * (1 + temp / 100), 0, 255)
            result[..., 0] = torch.clamp(result[..., 0] * (1 - temp / 200), 0, 255)

    if gamma != 1.0:
        gamma = max(0.01, gamma)
        inv_gamma = 1.0 / gamma
        result = torch.clamp((result / 255.0) ** inv_gamma * 255.0, 0, 255)

    if lut_file is not None:
        table = torch.tensor(read_3dl_lut(lut_file.name), device=device, dtype=torch.float32)
        result = apply_lut_torch(result, table, lut_intensity)

    return result.detach().cpu().numpy().astype(np.uint8)

def read_3dl_lut(file_path):
    """Read a .3dl LUT file and return the lookup table."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Remove comments and empty lines
    lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
    
    # Find LUT size
    size = None
    for line in lines:
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[-1])
            break
            
    if size is None:
        size = 32  # Default size
    
    # Get data lines (skip header lines)
    data_lines = []
    for line in lines:
        if not any(line.startswith(x) for x in ['TITLE', 'LUT_3D_SIZE']):
            try:
                r, g, b = map(float, line.split())
                data_lines.append([r, g, b])
            except:
                continue
    
    # Convert to numpy array
    table = np.array(data_lines)
    if len(table) == size * size * size:
        return table.reshape(size, size, size, 3)
    else:
        raise ValueError(f"Invalid LUT size. Expected {size*size*size} entries, got {len(table)}")

def apply_lut(image, lut_file, intensity=1.0):
    if image is None or lut_file is None:
        return None
    
    # Read LUT file
    table = read_3dl_lut(lut_file.name)
    size = table.shape[0]
    
    # Convert image to float32 (0-1 range)
    img = image.astype(np.float32) / 255.0
    
    # Reshape image for processing
    h, w, c = img.shape
    img_reshaped = img.reshape(-1, 3)
    
    # Get coordinates in LUT space
    coords = np.clip(img_reshaped * (size - 1), 0, size - 1)
    coords_floor = coords.astype(int)
    coords_ceil = np.minimum(coords_floor + 1, size - 1)
    
    # Get interpolation weights
    weights = coords - coords_floor
    
    # Get corner values
    result = np.zeros_like(img_reshaped)
    for i in range(img_reshaped.shape[0]):
        x, y, z = coords_floor[i]
        x1, y1, z1 = coords_ceil[i]
        wx, wy, wz = weights[i]
        
        # Get values at corners
        v000 = table[x, y, z]
        v001 = table[x, y, z1]
        v010 = table[x, y1, z]
        v011 = table[x, y1, z1]
        v100 = table[x1, y, z]
        v101 = table[x1, y, z1]
        v110 = table[x1, y1, z]
        v111 = table[x1, y1, z1]
        
        # Interpolate
        v00 = v000 * (1 - wx) + v100 * wx
        v01 = v001 * (1 - wx) + v101 * wx
        v10 = v010 * (1 - wx) + v110 * wx
        v11 = v011 * (1 - wx) + v111 * wx
        
        v0 = v00 * (1 - wy) + v10 * wy
        v1 = v01 * (1 - wy) + v11 * wy
        
        result[i] = v0 * (1 - wz) + v1 * wz
    
    # Reshape back
    result = result.reshape(h, w, 3)
    
    # Apply intensity blend
    if intensity < 1.0:
        result = img * (1 - intensity) + result * intensity
        
    # Convert to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def process_image(image, color_noise, mono_noise, gauss_noise, digital_grain,
                 blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file,
                 lut_intensity=0.5, use_gpu=False):
    if image is None:
        return None

    gpu_enabled, _ = gpu_available()
    if use_gpu and gpu_enabled:
        device = torch.device("cuda")
        return process_image_torch(
            np.array(image),
            color_noise,
            mono_noise,
            gauss_noise,
            digital_grain,
            blur_kernel,
            brightness,
            contrast,
            saturation,
            temperature,
            gamma,
            lut_file,
            lut_intensity,
            device,
        )

    result = np.array(image)

    # Apply different types of noise
    if color_noise > 0:
        result = apply_color_noise(result, color_noise)
    if mono_noise > 0:
        result = apply_monochrome_noise(result, mono_noise)
    if gauss_noise > 0:
        result = apply_gaussian_noise(result, gauss_noise)
    if digital_grain > 0:
        result = apply_digital_grain(result, digital_grain)

    if blur_kernel > 1:
        result = apply_gaussian_blur(result, blur_kernel)

    if brightness != 1.0:
        result = adjust_brightness(result, brightness)

    if contrast != 1.0:
        result = adjust_contrast(result, contrast)

    if saturation != 1.0:
        result = adjust_saturation(result, saturation)

    if temperature != 0:
        result = adjust_color_temperature(result, temperature)

    if gamma != 1.0:
        result = adjust_gamma(result, gamma)

    if lut_file is not None:
        result = apply_lut(result, lut_file, lut_intensity)

    return result

def reset_controls():
    """Reset all controls to their default values"""
    return [
        0,      # color_noise
        0,      # mono_noise
        0,      # gauss_noise
        0,      # digital_grain
        1,      # blur_kernel
        1.0,    # brightness
        1.0,    # contrast
        1.0,    # saturation
        0,      # temperature
        1.0,    # gamma
        0.5     # lut_intensity
    ]

def process_batch(input_dir, output_dir, color_noise, mono_noise, gauss_noise, digital_grain,
                blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file,
                lut_intensity=0.5, use_gpu=False):
    """Process all images in the input directory and save results to output directory"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    
    processed_count = 0
    for filename in image_files:
        try:
            # Read image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            logger.info(f"Processing file: {filename}")

            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process image
            result = process_image(img, color_noise, mono_noise, gauss_noise, digital_grain,
                                 blur_kernel, brightness, contrast, saturation, temperature,
                                 gamma, lut_file, lut_intensity, use_gpu)
            
            if result is not None:
                # Convert back to BGR for saving
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                # Save processed image
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, result)
                processed_count += 1
                logger.info(f"Finished file: {filename} -> {output_path}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue
    
    return f"Processed {processed_count} of {len(image_files)} images"

def save_current_settings(name, c_noise, m_noise, g_noise, d_grain, blur, bright, cont, sat, temp, gamma, lut_int):
    """Save current settings as a preset"""
    if not name.strip():
        return gr.Dropdown(choices=load_presets().keys())
    settings = {
        'color_noise': c_noise,
        'mono_noise': m_noise,
        'gauss_noise': g_noise,
        'digital_grain': d_grain,
        'blur': blur,
        'brightness': bright,
        'contrast': cont,
        'saturation': sat,
        'temperature': temp,
        'gamma': gamma,
        'lut_intensity': lut_int
    }
    preset_list = save_preset(name.strip(), settings)
    return gr.Dropdown(choices=preset_list)

def apply_preset_settings(preset_name):
    """Apply settings from a saved preset"""
    presets = load_presets()
    if preset_name in presets:
        settings = presets[preset_name]
        return [
            settings.get('color_noise', 0),
            settings.get('mono_noise', 0),
            settings.get('gauss_noise', 0),
            settings.get('digital_grain', 0),
            settings.get('blur', 1),
            settings.get('brightness', 1.0),
            settings.get('contrast', 1.0),
            settings.get('saturation', 1.0),
            settings.get('temperature', 0),
            settings.get('gamma', 1.0),
            settings.get('lut_intensity', 0.5)
        ]
    # Fallback to default control values when no preset is selected
    return [
        0,      # color_noise
        0,      # mono_noise
        0,      # gauss_noise
        0,      # digital_grain
        1,      # blur_kernel
        1.0,    # brightness
        1.0,    # contrast
        1.0,    # saturation
        0,      # temperature
        1.0,    # gamma
        0.5     # lut_intensity
    ]

# Create Gradio interface
with gr.Blocks(title="LUTplus - Image PostProcessing Tools") as demo:
    gr.Markdown("# LUTplus - Image PostProcessing Tools")
    gr.Markdown("Upload an image and apply various effects to it.")

    gpu_enabled, gpu_name = gpu_available()
    gpu_status = "✅ CUDA detected: {name}".format(name=gpu_name) if gpu_enabled else "⚠️ GPU disabled ({reason})".format(reason=gpu_name)
    use_gpu = gr.Checkbox(
        label="Use GPU (PyTorch + CUDA)",
        value=gpu_enabled,
        interactive=gpu_enabled,
        info="Enable acceleration when a CUDA-capable GPU and PyTorch are available."
    )
    gr.Markdown(gpu_status)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image")
            
            with gr.Accordion("Main Effects", open=True):
                blur_kernel = gr.Slider(minimum=1, maximum=31, value=1, step=2, label="Blur")
                brightness = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Brightness")
                contrast = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Contrast")
                saturation = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Saturation")
                temperature = gr.Slider(minimum=-100, maximum=100, value=0, step=1, label="Color Temperature")
                gamma = gr.Slider(minimum=0.2, maximum=5.0, value=1.0, step=0.1, label="Gamma")
            
            with gr.Accordion("Noise Effects", open=True):
                color_noise = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Color Noise")
                mono_noise = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Monochromatic Noise")
                gauss_noise = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Gaussian Noise")
                digital_grain = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Digital Grain")
        
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            process_btn = gr.Button("Process Image", variant="primary")
            
            with gr.Accordion("LUT Settings", open=True):
                lut_file = gr.File(label="LUT File (Optional)")
                lut_intensity = gr.Slider(minimum=0.0, maximum=0.5, value=0.5, step=0.025, label="LUT Intensity")
            
            with gr.Accordion("Batch Processing", open=True):
                input_dir = gr.Textbox(label="Input Directory", placeholder="Path to folder with images...")
                output_dir = gr.Textbox(label="Output Directory", placeholder="Path to save processed images...")
                process_batch_btn = gr.Button("Process Batch")
                batch_result = gr.Textbox(label="Batch Processing Result", interactive=False)
            
            with gr.Accordion("Presets", open=True):
                with gr.Row():
                    preset_name = gr.Textbox(label="Preset Name", placeholder="Enter preset name...")
                    save_preset_btn = gr.Button("Save Preset")
                
                with gr.Row():
                    preset_dropdown = gr.Dropdown(choices=load_presets().keys(), label="Presets", value=None)
                    apply_preset_btn = gr.Button("Apply Preset")
                    delete_preset_btn = gr.Button("Delete Preset")
            
            reset_btn = gr.Button("Reset All")
    
    def batch_process_wrapper(input_dir, output_dir, c_noise, m_noise, g_noise, d_grain, blur, bright,
                            cont, sat, temp, gamma, lut_file, lut_int, use_gpu_flag):
        if not input_dir or not output_dir:
            return "Please specify both input and output directories"
        if not os.path.exists(input_dir):
            return "Input directory does not exist"
        return process_batch(input_dir, output_dir, c_noise, m_noise, g_noise, d_grain,
                           blur, bright, cont, sat, temp, gamma, lut_file, lut_int, use_gpu_flag)
    
    save_preset_btn.click(
        fn=save_current_settings,
        inputs=[preset_name, color_noise, mono_noise, gauss_noise, digital_grain,
                blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_intensity],
        outputs=[preset_dropdown]
    )
    
    delete_preset_btn.click(
        fn=delete_preset,
        inputs=[preset_dropdown],
        outputs=[preset_dropdown]
    )
    
    apply_preset_btn.click(
        fn=apply_preset_settings,
        inputs=[preset_dropdown],
        outputs=[color_noise, mono_noise, gauss_noise, digital_grain,
                blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_intensity]
    )
    
    reset_btn.click(
        fn=reset_controls,
        inputs=[],
        outputs=[color_noise, mono_noise, gauss_noise, digital_grain,
                blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_intensity]
    )
    
    process_btn.click(
        fn=process_image,
        inputs=[input_image, color_noise, mono_noise, gauss_noise, digital_grain,
                blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity, use_gpu],
        outputs=output_image
    )

    process_batch_btn.click(
        fn=batch_process_wrapper,
        inputs=[input_dir, output_dir, color_noise, mono_noise, gauss_noise, digital_grain,
                blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity, use_gpu],
        outputs=batch_result
    )

if __name__ == "__main__":
    print("\nLUTplus is starting...")

    # Check for --network argument
    is_network_mode = "--network" in sys.argv
    is_gpu_mode = "--gpu" in sys.argv
    if is_gpu_mode:
        GPU_ENABLED = _is_gpu_available()
        if GPU_ENABLED:
            logger.info("GPU mode requested: CUDA device detected and will be used where possible.")
        else:
            logger.warning("GPU mode requested but no CUDA-enabled OpenCV build was found. Falling back to CPU.")
    server_ip = "0.0.0.0" if is_network_mode else "127.0.0.1"
    
    if is_network_mode:
        try:
            # Better method to get local IP address
            import socket
            def get_local_ip():
                # Get all network interfaces that might be accessible from other devices
                possible_ips = []
                
                # Try to get a list of all network interfaces
                try:
                    # Create a socket that connects to an external address
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    # This doesn't actually send any packets
                    s.connect(("8.8.8.8", 80))
                    # Get the IP address used for this connection
                    local_ip = s.getsockname()[0]
                    s.close()
                    possible_ips.append(local_ip)
                except:
                    pass
                
                # If the above method failed, try listing all interfaces
                if not possible_ips:
                    try:
                        hostname = socket.gethostname()
                        # Get all IPs for this hostname
                        ips = socket.getaddrinfo(hostname, None)
                        for ip in ips:
                            if ip[0] == socket.AF_INET:  # Only IPv4 addresses
                                addr = ip[4][0]
                                # Skip localhost
                                if not addr.startswith('127.'):
                                    possible_ips.append(addr)
                    except:
                        pass
                
                # If we still don't have any IPs, return a message
                if not possible_ips:
                    return "unknown (check your network)"
                
                # Prefer 192.168.x.x or 10.x.x.x addresses (common for LANs)
                for ip in possible_ips:
                    if ip.startswith('192.168.') or ip.startswith('10.'):
                        return ip
                
                # Otherwise just return the first IP
                return possible_ips[0]
            
            local_ip = get_local_ip()
            print(f"Network mode enabled. Access from local network: http://{local_ip}:7860")
        except:
            print("Network mode enabled. Application will be accessible from your local network.")
    
    try:
        # Launch the application
        demo.launch(
            quiet=True,  # Disable Gradio messages, including version warning
            show_error=True,
            show_api=False,
            server_name=server_ip,  # Use 0.0.0.0 for network access or 127.0.0.1 for local only
            inbrowser=True,  # Enable automatic browser launch
            share=False  # Disable public URL
        )
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again.")
        input("Press Enter to exit...")
        sys.exit(1)