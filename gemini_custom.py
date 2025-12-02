import time
import os
import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types

# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def tensor_to_pil(tensor_image):
    """Converts ComfyUI Tensor to PIL Images."""
    batch_results = []
    if tensor_image is None: return []
    for i in range(tensor_image.shape[0]):
        img = tensor_image[i]
        img = 255. * img.cpu().numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        batch_results.append(Image.fromarray(img))
    return batch_results

def pil_to_tensor(pil_images):
    """Converts PIL Images to ComfyUI Tensor."""
    tensor_list = []
    for img in pil_images:
        img = img.convert("RGB")
        img_array = np.array(img).astype(np.float32) / 255.0
        tensor_list.append(torch.from_numpy(img_array))
    if not tensor_list: return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    return torch.stack(tensor_list, dim=0)

def get_client(api_key_input):
    """Securely creates the Google GenAI Client."""
    key = api_key_input.strip()
    if not key: key = os.getenv("GOOGLE_API_KEY")
    if not key: key = os.getenv("GEMINI_API_KEY")
    if not key: raise ValueError("API Key missing. Set GOOGLE_API_KEY env var or fill the widget.")
    return genai.Client(api_key=key)

def get_safety_config(mode):
    """Returns the safety settings list based on user selection."""
    if mode == "Standard": return None
    threshold = "BLOCK_NONE" if mode == "Block None (Uncensored)" else "BLOCK_LOW_AND_ABOVE"
    return [
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold=threshold),
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold=threshold),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold=threshold),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold=threshold),
    ]

# --------------------------------------------------------------------------------
# NODE 1: GEMINI CHAT & VISION (UPDATED WITH SYSTEM PROMPT)
# --------------------------------------------------------------------------------

class GeminiCustomNode:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image"}),
                "model": (["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-pro-exp-02-05"],),
                "safety_level": (["Block None (Uncensored)", "Standard", "Strict"], {"default": "Standard"}),
            },
            "optional": {
                "system_instruction": ("STRING", {
                    "multiline": True, 
                    "default": "", 
                    "tooltip": "Define the model's persona, role, or constraints (e.g., 'You are a senior coder')."
                }),
                "image": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate"
    CATEGORY = "Google/Gemini"
    DESCRIPTION = "Chat & Vision with System Instructions."

    def generate(self, prompt, model, safety_level, system_instruction="", image=None, api_key=""):
        try:
            client = get_client(api_key)
            contents = [prompt]
            if image is not None: contents.extend(tensor_to_pil(image))
            
            # Add System Instruction to Config
            config_args = {
                "temperature": 1.0,
                "safety_settings": get_safety_config(safety_level)
            }
            
            if system_instruction and system_instruction.strip():
                config_args["system_instruction"] = system_instruction.strip()

            config = types.GenerateContentConfig(**config_args)
            
            response = client.models.generate_content(model=model, contents=contents, config=config)
            return (response.text,)
        except Exception as e: return (f"Error: {str(e)}",)

# --------------------------------------------------------------------------------
# NODE 2: GEMINI IMAGE GEN (NANO BANANA)
# --------------------------------------------------------------------------------

class GeminiImageGenNode:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic city"}),
                "model": (["gemini-2.5-flash-image", "gemini-3-pro-image-preview"],),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "1:1"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "safety_level": (["Block None (Uncensored)", "Standard", "Strict"], {"default": "Standard"}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text_response")
    FUNCTION = "generate_image"
    CATEGORY = "Google/Gemini"
    DESCRIPTION = "Generates images using Nano Banana models."

    def generate_image(self, prompt, model, aspect_ratio, resolution, safety_level, reference_images=None, api_key=""):
        try:
            client = get_client(api_key)
            contents = [prompt]
            if reference_images is not None: contents.extend(tensor_to_pil(reference_images))
            
            config = types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=resolution),
                safety_settings=get_safety_config(safety_level)
            )
            response = client.models.generate_content(model=model, contents=contents, config=config)
            
            pil_imgs = []
            text_out = []
            if response.parts:
                for part in response.parts:
                    if part.text: text_out.append(part.text)
                    if hasattr(part, "as_image") and part.as_image(): pil_imgs.append(part.as_image())
            
            final_tensor = pil_to_tensor(pil_imgs) if pil_imgs else torch.zeros((1, 64, 64, 3))
            return (final_tensor, "\n".join(text_out))
        except Exception as e: return (torch.zeros((1, 64, 64, 3)), f"Error: {str(e)}")

# --------------------------------------------------------------------------------
# NODE 3: GEMINI TTS (SPEECH)
# --------------------------------------------------------------------------------

class GeminiTTSNode:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        VOICES = [
            "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe",
            "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina",
            "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam",
            "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix",
            "Sadachbia", "Sadaltager", "Sulafat", "Zephyr"
        ]
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Say something..."}),
                "model": (["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],),
                "voice": (sorted(VOICES), {"default": "Puck"}),
                "safety_level": (["Block None (Uncensored)", "Standard", "Strict"], {"default": "Standard"}),
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "Google/Gemini"
    DESCRIPTION = "Generates Speech."

    def generate_speech(self, text, model, voice, safety_level, api_key=""):
        try:
            client = get_client(api_key)
            config = types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                    )
                ),
                safety_settings=get_safety_config(safety_level)
            )
            response = client.models.generate_content(model=model, contents=text, config=config)

            audio_data = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        audio_data = part.inline_data.data
                        break
            if not audio_data: raise ValueError("No audio data found.")

            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
            return ({"waveform": audio_tensor, "sample_rate": 24000},)

        except Exception as e:
            print(f"TTS Error: {e}")
            return ({"waveform": torch.zeros((1, 1, 100)), "sample_rate": 24000},)

# --------------------------------------------------------------------------------
# NODE 4: VEO ADVANCED (VIDEO)
# --------------------------------------------------------------------------------

class VeoAdvancedNode:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot..."}),
                "model": (["veo-3.1-generate-preview", "veo-3.0-generate-001"],),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "reference_images": ("IMAGE",),
                "video_to_extend": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate_video"
    CATEGORY = "Google/Veo"
    DESCRIPTION = "Generates Video. Req: Google AI Pro/Ultra."

    def generate_video(self, prompt, model, aspect_ratio, resolution, negative_prompt="", 
                       start_image=None, end_image=None, reference_images=None, 
                       video_to_extend="", api_key=""):
        try:
            client = get_client(api_key)
            config_args = {"aspect_ratio": aspect_ratio, "resolution": resolution}
            if negative_prompt.strip(): config_args["negative_prompt"] = negative_prompt.strip()
            if end_image is not None:
                p = tensor_to_pil(end_image)
                if p: config_args["last_frame"] = p[0]
            if reference_images is not None:
                pr = tensor_to_pil(reference_images)
                config_args["reference_images"] = [types.VideoGenerationReferenceImage(image=i, reference_type="asset") for i in pr]

            video_config = types.GenerateVideosConfig(**config_args)
            kwargs = {"model": model, "prompt": prompt, "config": video_config}
            
            if start_image is not None:
                p = tensor_to_pil(start_image)
                if p: kwargs["image"] = p[0]
            
            if video_to_extend and os.path.exists(video_to_extend):
                print("Uploading video...")
                vid_ref = client.files.upload(path=video_to_extend)
                while vid_ref.state.name == "PROCESSING":
                    time.sleep(2)
                    vid_ref = client.files.get(name=vid_ref.name)
                kwargs["video"] = vid_ref

            print(f"Starting Veo generation ({model})...")
            operation = client.models.generate_videos(**kwargs)
            while not operation.done:
                print("Waiting for Veo...")
                time.sleep(10)
                operation = client.operations.get(operation)

            if not operation.response.generated_videos: return ("Error: No video",)
            
            out_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(out_dir, exist_ok=True)
            fname = f"veo_{int(time.time())}.mp4"
            fpath = os.path.join(out_dir, fname)
            client.files.download(file=operation.response.generated_videos[0].video, path=fpath)
            return (fpath,)

        except Exception as e: return (f"Error: {str(e)}",)

# --------------------------------------------------------------------------------
# MAPPINGS
# --------------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GeminiCustomNode": GeminiCustomNode,
    "GeminiImageGenNode": GeminiImageGenNode,
    "GeminiTTSNode": GeminiTTSNode,
    "VeoAdvancedNode": VeoAdvancedNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiCustomNode": "Google Gemini (Chat/Vision)",
    "GeminiImageGenNode": "Google Gemini Image (Nano Banana)",
    "GeminiTTSNode": "Google Gemini TTS (Speech)",
    "VeoAdvancedNode": "Google Veo (Video)"
}
