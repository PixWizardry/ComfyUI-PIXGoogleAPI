# ComfyUI-PIXGoogleAPI

A secure, local, and fully-featured suite of Custom Nodes for ComfyUI that connects directly to Google's latest AI models via the official `google-genai` SDK (v2).

**Supports:** Gemini 2.0/2.5 (Chat & Vision), Nano Banana (Image Gen), Gemini TTS (Speech), and Veo (Video).

## 🚀 Key Features

*   **Zero-Leak Security:** Designed for privacy. Your API Key is **never** saved in image metadata or workflow (`.json`) files.
*   **Direct Connection:** Connects directly from your machine to Google Cloud. No third-party proxies, no cloud middleman.
*   **Latest Models:** Support for Gemini 2.5, Gemini 2.0 Pro Experimental, and the new SDK architecture.
*   **Safety Control:** "Safety Settings" dropdown allows you to set filters to `Block None` for maximum creative freedom.
    *   Gemini API has built-in protections against core, and cannot be adjusted.
*   **Native Integration:**
    *   Images handle `Batch` inputs automatically.
    *   Audio outputs standard ComfyUI `AUDIO` tensors (compatible with VHS/Save Audio).
    *   Video outputs standard `mp4` files.

---

## 📦 Installation

1.  Open your command line/terminal.
2.  Navigate to your ComfyUI custom nodes directory:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
3.  Clone this repository:
    ```bash
    git clone https://github.com/PixWizardry/ComfyUI-PIXGoogleAPI.git
    ```
4.  Install the required Google SDK:
    ```bash
    pip install -r ComfyUI-PIXGoogleAPI/requirements.txt
    ```
    *Note: You must restart ComfyUI after installing.*

---

## 🔑 Setup & Authentication (Important)

To keep your API Key secure, it is highly recommended to use **Environment Variables**. As Recommended By Google Gemini SDK Documentation.

1.  Get your API Key from [Google AI Studio](https://aistudio.google.com/).
2.  **Windows:**
    *   Press `Win + R`, type `cmd`, and press Enter.
    *   Run: `setx GOOGLE_API_KEY "your_actual_api_key_here"`
    *   Restart ComfyUI (and your terminal).

**Usage in ComfyUI:**
*   Leave the `api_key` widget in the nodes **EMPTY**. The node will automatically find your system key.
*   *Optional:* You can type a key into the widget for temporary testing, but this is less secure as it saves the key inside your workflow file. ALWAYS DELETE IT, Especially if you share your workflow!!!

---

## 🧩 Node Guide

### 1. Google Gemini (Chat/Vision)
A multimodal LLM node.
*   **Inputs:** Text Prompt + Optional Image(s).
*   **Models:** `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.0-pro-exp`.
*   **Usage:** Image captioning, visual question answering, or standard chatting.
*   **Safety:** Use "Block None" to help the model from refusing prompts.

### 2. Google Gemini Image (Nano Banana)
State-of-the-art image generation and editing.
*   **Models:**
    *   `gemini-2.5-flash-image` (Fast/High Volume)
    *   `gemini-3-pro-image-preview` (High Fidelity/Reasoning)
*   **Modes:**
    *   **Text-to-Image:** Enter a prompt, get an image.
    *   **Image Editing:** Connect an image to `reference_images` and prompt the change (e.g., "Change the cat to a dog").
*   **Outputs:** Returns the Image + the Text reasoning (Pro model explains *why* it drew the image).

### 3. Google Gemini TTS (Speech)
High-quality, 24kHz text-to-speech generation.
*   **Voices:** Over 30 voices (Puck, Fenrir, Kore, etc.).
*   **Style Control:** You can use natural language in the prompt to direct the acting.
    *   *Example:* "Say this in a whispered, terrified voice: 'Did you hear that?'"
*   **Output:** Standard ComfyUI Audio format.

### 4. Google Veo (Video)
**⚠️ Requirements:** Requires a paid **Google API** subscription.
*   **Features:**
    *   **Text-to-Video:** Cinematic generation.
    *   **Image-to-Video:** Connect `start_image` to animate a static picture.
    *   **Steering:** Connect `start_image` AND `end_image` to morph between two states.
    *   **Reference:** Connect `reference_images` (batch supported) to maintain character/style consistency.
    *   **Extend:** Paste a local file path to an .mp4 to extend the footage.
*   **Note:** This node uses asynchronous polling. It may take 60+ seconds to generate. Check your console for status updates.

  ### 💡 Advanced: System Instructions
The **Google Gemini (Chat/Vision)** node now supports **System Instructions** (also known as System Prompts).

This input allows you to define the behavior, persona, and constraints of the model *before* it answers your prompt.

**Example Use Cases:**
*   **Persona:** "You are a cynical film noir detective. Answer all questions in a gritty monologue."
*   **Format:** "You are a Python coding assistant. Return only raw code. Do not wrap in markdown blocks. Do not explain the code."
*   **Constraints:** "Answer in JSON format only. Fields: 'title', 'summary'."

---

## ⚠️ Troubleshooting

**Error: "403 Forbidden" or "Billing not enabled"**
*   This usually happens with the **Veo Node**. Veo is a paid feature. Ensure your Google Cloud Project linked to the API Key has billing enabled and you are subscribed to the appropriate tier.

**Error: "API Key not found"**
*   Ensure you restarted ComfyUI after setting the Environment Variable.
*   Ensure you didn't accidentally paste the key with spaces.

---
