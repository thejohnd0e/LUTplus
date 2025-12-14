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

# GPU support with CuPy
USE_GPU = False
try:
    import cupy as cp
    USE_GPU = True
    print("GPU support enabled: CuPy detected")
except ImportError:
    print("GPU support disabled: CuPy not available. Install cupy-cuda12x or cupy-cuda11x for GPU acceleration.")
    cp = None

# Set GPU mode from command line
if "--gpu" in sys.argv:
    if USE_GPU:
        print("GPU mode: ENABLED")
    else:
        print("Warning: --gpu flag specified but CuPy is not available. Falling back to CPU.")
        USE_GPU = False

# Disable Gradio update checks
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["GRADIO_TELEMETRY_ENABLED"] = "False"
os.environ["GRADIO_SKIP_VERSION_CHECK"] = "True"

# Pydantic compatibility fix
try:
    import pydantic
    import fastapi
    from fastapi import FastAPI
    
    pydantic_version = getattr(pydantic, '__version__', '0')
    
    if pydantic_version.startswith('2.'):
        if not hasattr(pydantic, 'RootModel'):
            class RootModelV1Compat(pydantic.BaseModel):
                root: object
                def __init__(self, root=None, **kwargs):
                    super().__init__(root=root, **kwargs)
                @property
                def model_dump(self):
                    return lambda *args, **kwargs: {'root': self.root}
            setattr(pydantic, 'RootModel', RootModelV1Compat)
        
        orig_fastapi_init = fastapi.FastAPI.__init__
        def patched_fastapi_init(self, *args, **kwargs):
            if 'model_config' not in kwargs:
                kwargs['model_config'] = {'arbitrary_types_allowed': True}
            elif isinstance(kwargs['model_config'], dict):
                kwargs['model_config']['arbitrary_types_allowed'] = True
            return orig_fastapi_init(self, *args, **kwargs)
        fastapi.FastAPI.__init__ = patched_fastapi_init
        
        if hasattr(fastapi, '_compat'):
            if hasattr(fastapi._compat, 'get_model_fields'):
                orig_get_model_fields = fastapi._compat.get_model_fields
                def safe_get_model_fields(model):
                    try:
                        from starlette.requests import Request
                        if model == Request or getattr(model, '__name__', '') == 'Request':
                            return {}
                        return orig_get_model_fields(model)
                    except Exception:
                        return {}
                fastapi._compat.get_model_fields = safe_get_model_fields
    else:
        pass
        
    orig_import = __import__
    def custom_import(name, *args, **kwargs):
        if name == 'pydantic' and 'BaseSettings' in args[2]:
            if not hasattr(sys, '_showed_basesettings_warning'):
                print("Info: BaseSettings has been moved to pydantic-settings package in Pydantic v2")
                setattr(sys, '_showed_basesettings_warning', True)
        return orig_import(name, *args, **kwargs)
except Exception as e:
    import traceback
    print(f"Warning: failed to apply Pydantic patch: {e}")
    print(traceback.format_exc())

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])
logging.getLogger('gradio').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('fastapi').setLevel(logging.WARNING)
logging.getLogger('pydantic').setLevel(logging.WARNING)

PRESETS_FILE = "presets.json"

# Helper functions for GPU/CPU array conversion
def to_gpu_array(arr):
    """Convert numpy array to cupy array if GPU is enabled"""
    if USE_GPU and cp is not None:
        return cp.asarray(arr)
    return arr

def to_cpu_array(arr):
    """Convert cupy array to numpy array if needed"""
    if USE_GPU and cp is not None and hasattr(arr, 'get'):
        return arr.get()
    return arr

def get_array_module(arr):
    """Get the appropriate array module (cupy or numpy)"""
    if USE_GPU and cp is not None:
        return cp.get_array_module(arr) if hasattr(arr, '__cuda_array_interface__') or (hasattr(cp, 'get_array_module') and isinstance(arr, cp.ndarray)) else np
    return np

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
        return [settings.get('noise_level', 0), settings.get('blur', 1), settings.get('brightness', 1.0), settings.get('saturation', 1.0), settings.get('temperature', 0), settings.get('lut_intensity', 0.5)]
    return [0, 1, 1.0, 1.0, 0, 0.5]

def apply_color_noise(image, level):
    if level == 0:
        return image
    xp = cp if USE_GPU and cp is not None else np
    img = to_gpu_array(image.astype(np.float32))
    noise = xp.random.normal(0, level, img.shape).astype(xp.float32)
    noisy_img = img + noise
    result = xp.clip(noisy_img, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def apply_monochrome_noise(image, level):
    if level == 0:
        return image
    xp = cp if USE_GPU and cp is not None else np
    img = to_gpu_array(image.astype(np.float32))
    noise = xp.random.normal(0, level, image.shape[:2]).astype(xp.float32)
    noise_3ch = xp.stack([noise] * 3, axis=-1)
    noisy_img = img + noise_3ch
    result = xp.clip(noisy_img, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def apply_gaussian_noise(image, level):
    if level == 0:
        return image
    xp = cp if USE_GPU and cp is not None else np
    img = to_gpu_array(image.astype(np.float32))
    noise = xp.random.normal(0, level, img.shape).astype(xp.float32)
    noisy_img = img + noise
    result = xp.clip(noisy_img, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def apply_digital_grain(image, level):
    if level == 0:
        return image
    xp = cp if USE_GPU and cp is not None else np
    img = to_gpu_array(image.astype(np.float32))
    grain = xp.random.uniform(-level, level, img.shape[:2]).astype(xp.float32)
    grain = xp.where(grain > level/2, level, -level)
    grain_color = xp.stack([grain, grain * 0.5, grain * 0.5], axis=-1)
    noisy_img = img + grain_color
    result = xp.clip(noisy_img, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def apply_gaussian_blur(image, kernel_size):
    if image is None:
        return None
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def adjust_brightness(image, factor):
    if image is None:
        return None
    xp = cp if USE_GPU and cp is not None else np
    img = to_gpu_array(image.astype(xp.float32))
    result = img * factor
    result = xp.clip(result, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def adjust_saturation(image, factor):
    if image is None:
        return None
    # OpenCV operations stay on CPU, but we can use GPU for array operations
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    xp = cp if USE_GPU and cp is not None else np
    hsv_gpu = to_gpu_array(hsv)
    hsv_gpu[:, :, 1] = hsv_gpu[:, :, 1] * factor
    hsv_gpu[:, :, 1] = xp.clip(hsv_gpu[:, :, 1], 0, 255)
    hsv = to_cpu_array(hsv_gpu).astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return result

def adjust_color_temperature(image, temperature):
    if image is None:
        return None
    xp = cp if USE_GPU and cp is not None else np
    result = to_gpu_array(image.copy())
    if temperature > 0:
        result[:, :, 0] = xp.clip(result[:, :, 0] * (1 + temperature/100), 0, 255)
        result[:, :, 2] = xp.clip(result[:, :, 2] * (1 - temperature/200), 0, 255)
    else:
        temperature = abs(temperature)
        result[:, :, 2] = xp.clip(result[:, :, 2] * (1 + temperature/100), 0, 255)
        result[:, :, 0] = xp.clip(result[:, :, 0] * (1 - temperature/200), 0, 255)
    return to_cpu_array(result.astype(xp.uint8))

def adjust_gamma(image, gamma):
    if image is None or gamma == 1.0:
        return image
    gamma = max(0.01, gamma)
    inv_gamma = 1.0 / gamma
    xp = cp if USE_GPU and cp is not None else np
    table = xp.array([((i / 255.0) ** inv_gamma) * 255 for i in xp.arange(0, 256)]).astype(xp.uint8)
    table_cpu = to_cpu_array(table) if USE_GPU else table
    return cv2.LUT(image, table_cpu)

def adjust_contrast(image, factor):
    if image is None or factor == 1.0:
        return image
    xp = cp if USE_GPU and cp is not None else np
    img = to_gpu_array(image.astype(xp.float32))
    mean = float(xp.mean(img))
    adjusted = img * factor + mean * (1 - factor)
    result = xp.clip(adjusted, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def read_3dl_lut(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip() and not l.startswith('#')]
    size = None
    for line in lines:
        if line.startswith('LUT_3D_SIZE'):
            size = int(line.split()[-1])
            break
    if size is None:
        size = 32
    data_lines = []
    for line in lines:
        if not any(line.startswith(x) for x in ['TITLE', 'LUT_3D_SIZE']):
            try:
                r, g, b = map(float, line.split())
                data_lines.append([r, g, b])
            except:
                continue
    table = np.array(data_lines)
    if len(table) == size * size * size:
        return table.reshape(size, size, size, 3)
    else:
        raise ValueError(f"Invalid LUT size. Expected {size*size*size} entries, got {len(table)}")

def apply_lut(image, lut_file, intensity=1.0):
    if image is None or lut_file is None:
        return None
    xp = cp if USE_GPU and cp is not None else np
    table = read_3dl_lut(lut_file.name)
    table_gpu = to_gpu_array(table) if USE_GPU else table
    size = table.shape[0]
    img = to_gpu_array(image.astype(np.float32) / 255.0)
    h, w, c = img.shape
    img_reshaped = img.reshape(-1, 3)
    coords = xp.clip(img_reshaped * (size - 1), 0, size - 1)
    coords_floor = coords.astype(xp.int32)
    coords_ceil = xp.minimum(coords_floor + 1, size - 1)
    weights = coords - coords_floor.astype(xp.float32)
    
    # Vectorized trilinear interpolation
    x, y, z = coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2]
    x1, y1, z1 = coords_ceil[:, 0], coords_ceil[:, 1], coords_ceil[:, 2]
    wx, wy, wz = weights[:, 0], weights[:, 1], weights[:, 2]
    
    # Get all 8 corner values using advanced indexing
    # For 3D LUT: table[r, g, b] where r, g, b are indices
    v000 = table_gpu[x, y, z]
    v001 = table_gpu[x, y, z1]
    v010 = table_gpu[x, y1, z]
    v011 = table_gpu[x, y1, z1]
    v100 = table_gpu[x1, y, z]
    v101 = table_gpu[x1, y, z1]
    v110 = table_gpu[x1, y1, z]
    v111 = table_gpu[x1, y1, z1]
    
    # Trilinear interpolation
    wx_expanded = wx[:, xp.newaxis] if len(wx.shape) == 1 else wx
    wy_expanded = wy[:, xp.newaxis] if len(wy.shape) == 1 else wy
    wz_expanded = wz[:, xp.newaxis] if len(wz.shape) == 1 else wz
    
    v00 = v000 * (1 - wx_expanded) + v100 * wx_expanded
    v01 = v001 * (1 - wx_expanded) + v101 * wx_expanded
    v10 = v010 * (1 - wx_expanded) + v110 * wx_expanded
    v11 = v011 * (1 - wx_expanded) + v111 * wx_expanded
    
    v0 = v00 * (1 - wy_expanded) + v10 * wy_expanded
    v1 = v01 * (1 - wy_expanded) + v11 * wy_expanded
    
    result = v0 * (1 - wz_expanded) + v1 * wz_expanded
    result = result.reshape(h, w, 3)
    
    if intensity < 1.0:
        result = img * (1 - intensity) + result * intensity
    result = xp.clip(result * 255, 0, 255).astype(xp.uint8)
    return to_cpu_array(result)

def process_image(image, color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity=0.5):
    if image is None:
        return None
    print("Processing single image...")
    start_time = time.time()
    result = np.array(image)
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
    elapsed_time = time.time() - start_time
    print(f"Image processed in {elapsed_time:.2f} seconds")
    return result

def reset_controls():
    return [0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 0, 1.0, 0.5]

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def process_batch(input_dir, output_dir, color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f.lower())[1] in image_extensions]
    total_files = len(image_files)
    if total_files == 0:
        return "No images found in input directory"
    print(f"\n{'='*60}")
    print(f"Starting batch processing: {total_files} images")
    print(f"{'='*60}")
    processed_count = 0
    failed_count = 0
    batch_start_time = time.time()
    for idx, filename in enumerate(image_files, 1):
        try:
            file_start_time = time.time()
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{idx}/{total_files}] Failed to read: {filename}")
                failed_count += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = process_image(img, color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity)
            if result is not None:
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, result)
                processed_count += 1
                elapsed_time = time.time() - batch_start_time
                avg_time_per_image = elapsed_time / processed_count
                remaining_images = total_files - idx
                eta_seconds = avg_time_per_image * remaining_images
                file_time = time.time() - file_start_time
                progress_percent = (idx / total_files) * 100
                print(f"[{idx}/{total_files}] {filename}")
                print(f"  Progress: {progress_percent:.1f}% | Time: {file_time:.2f}s | ETA: {format_time(eta_seconds)}")
            else:
                print(f"[{idx}/{total_files}] Failed to process: {filename}")
                failed_count += 1
        except Exception as e:
            print(f"[{idx}/{total_files}] Error processing {filename}: {str(e)}")
            failed_count += 1
            continue
    total_time = time.time() - batch_start_time
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count}/{total_files} images")
    if failed_count > 0:
        print(f"Failed: {failed_count} images")
    print(f"Total time: {format_time(total_time)}")
    if processed_count > 0:
        print(f"Average time per image: {total_time/processed_count:.2f}s")
    print(f"{'='*60}\n")
    return f"Processed {processed_count} of {total_files} images (Failed: {failed_count}) in {format_time(total_time)}"

def save_current_settings(name, c_noise, m_noise, g_noise, d_grain, blur, bright, cont, sat, temp, gamma, lut_int):
    if not name.strip():
        return gr.Dropdown(choices=load_presets().keys())
    settings = {'color_noise': c_noise, 'mono_noise': m_noise, 'gauss_noise': g_noise, 'digital_grain': d_grain, 'blur': blur, 'brightness': bright, 'contrast': cont, 'saturation': sat, 'temperature': temp, 'gamma': gamma, 'lut_intensity': lut_int}
    preset_list = save_preset(name.strip(), settings)
    return gr.Dropdown(choices=preset_list)

def apply_preset_settings(preset_name):
    presets = load_presets()
    if preset_name in presets:
        settings = presets[preset_name]
        return [settings.get('color_noise', 0), settings.get('mono_noise', 0), settings.get('gauss_noise', 0), settings.get('digital_grain', 0), settings.get('blur', 1), settings.get('brightness', 1.0), settings.get('contrast', 1.0), settings.get('saturation', 1.0), settings.get('temperature', 0), settings.get('gamma', 1.0), settings.get('lut_intensity', 0.5)]
    return [0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 0, 1.0, 0.5]

def get_local_ip():
    possible_ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        possible_ips.append(local_ip)
    except:
        pass
    if not possible_ips:
        try:
            hostname = socket.gethostname()
            ips = socket.getaddrinfo(hostname, None)
            for ip in ips:
                if ip[0] == socket.AF_INET:
                    addr = ip[4][0]
                    if not addr.startswith('127.'):
                        possible_ips.append(addr)
        except:
            pass
    if not possible_ips:
        return "unknown"
    for ip in possible_ips:
        if ip.startswith('192.168.') or ip.startswith('10.'):
            return ip
    return possible_ips[0]

with gr.Blocks(title="LUTplus - Image PostProcessing Tools") as demo:
    gr.Markdown("# LUTplus - Image PostProcessing Tools")
    gr.Markdown("Upload an image and apply various effects to it.")
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
    
    def batch_process_wrapper(input_dir, output_dir, c_noise, m_noise, g_noise, d_grain, blur, bright, cont, sat, temp, gamma, lut_file, lut_int):
        if not input_dir or not output_dir:
            return "Please specify both input and output directories"
        if not os.path.exists(input_dir):
            return "Input directory does not exist"
        return process_batch(input_dir, output_dir, c_noise, m_noise, g_noise, d_grain, blur, bright, cont, sat, temp, gamma, lut_file, lut_int)
    
    save_preset_btn.click(fn=save_current_settings, inputs=[preset_name, color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_intensity], outputs=[preset_dropdown])
    delete_preset_btn.click(fn=delete_preset, inputs=[preset_dropdown], outputs=[preset_dropdown])
    apply_preset_btn.click(fn=apply_preset_settings, inputs=[preset_dropdown], outputs=[color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_intensity])
    reset_btn.click(fn=reset_controls, inputs=[], outputs=[color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_intensity])
    process_btn.click(fn=process_image, inputs=[input_image, color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity], outputs=output_image)
    process_batch_btn.click(fn=batch_process_wrapper, inputs=[input_dir, output_dir, color_noise, mono_noise, gauss_noise, digital_grain, blur_kernel, brightness, contrast, saturation, temperature, gamma, lut_file, lut_intensity], outputs=batch_result)

if __name__ == "__main__":
    print("\nLUTplus is starting...")
    if USE_GPU:
        print("GPU acceleration: ENABLED")
        if cp is not None:
            try:
                mempool = cp.get_default_memory_pool()
                print(f"GPU Memory: {mempool.get_limit() / 1024**3:.2f} GB available")
            except:
                pass
    else:
        print("GPU acceleration: DISABLED (CPU mode)")
    is_network_mode = "--network" in sys.argv
    server_ip = "0.0.0.0" if is_network_mode else "127.0.0.1"
    if is_network_mode:
        try:
            local_ip = get_local_ip()
            print(f"Network mode enabled. Access from local network: http://{local_ip}:7860")
        except:
            print("Network mode enabled. Application will be accessible from your local network.")
    try:
        demo.launch(quiet=True, show_error=True, show_api=False, server_name=server_ip, inbrowser=True, share=False)
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again.")
        input("Press Enter to exit...")
        sys.exit(1)