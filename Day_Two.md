***

## Day 2 — Audio, Speech Recognition, and Multi-Modal AI Models

### 1. Introduction to Speech Recognition and Whisper Models

Speech Recognition is the task of converting spoken audio into text. It powers many applications such as virtual assistants (Siri, Alexa), subtitles, call center automation, and accessibility tools.

**Whisper Models by OpenAI:**
- General-purpose speech recognition models trained on 680,000+ hours of diverse audio.
- Supports ~99 languages for transcription and even speech translation (e.g., Hindi audio → English text).
- Robust to background noise, accents, and specialized vocabulary.
- Available in various sizes (tiny, base, small, medium, large) balancing speed and accuracy.
- Open-source and can run locally without API costs.

***

### 2. Using Whisper Models for Speech Recognition

**Setup and Installation:**
```bash
pip install git+https://github.com/openai/whisper.git
pip install torch soundfile
```

**Basic Python Usage:**
```python
import whisper
model = whisper.load_model("small")
result = model.transcribe("audio_file.mp3")
print("Transcription:", result["text"])
```

Supports multiple audio formats like WAV, MP3, and M4A.

***

### 3. Multilingual Transcription and Translation with Whisper

- Detect the audio language automatically.
- Translate speech into English directly by specifying `task="translate"`.

Example:
```python
result = model.transcribe("hindi_audio.mp3", task="translate")
print("Translated to English:", result["text"])
```

***

### 4. Whisper via Hugging Face Transformers (Free Alternative)

Hugging Face provides free local inference of Whisper models without needing paid OpenAI API keys.

**Installation:**
```bash
pip install transformers datasets soundfile librosa
```

**Example:**
```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
result = pipe("sample_audio.wav")
print("Transcription:", result["text"])
```

Supports real-time-like recognition by processing short audio chunks recorded from the microphone.

***

### 5. Real-Time Speech Recognition Example

Using Python's `sounddevice` package to record audio and transcribe in chunks:

```python
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

def record_and_transcribe(duration=5, samplerate=16000):
    print("Recording... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    wav.write("temp.wav", samplerate, (audio * 32767).astype(np.int16))
    result = asr("temp.wav")
    print("Transcription:", result["text"])

while True:
    record_and_transcribe()
```

***

### 6. Multi-Task Speech Recognition and Translation Pipelines

Supports:
- Multilingual speech transcription (same language).
- Non-English to English translation.
- English to French translation.
- English to Hindi translation.

Using Hugging Face pipelines:
```python
asr_multilingual = pipeline("automatic-speech-recognition", model="openai/whisper-small")
asr_translate_to_en = pipeline("automatic-speech-recognition", model="openai/whisper-small", generate_kwargs={"task":"translate"})
translator_en_to_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
translator_en_to_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
```

***

### 7. Troubleshooting and Environment Setup

- Ubuntu's PEP 668 external environment restrictions require virtual environments (`venv`) for package installations.
- PyTorch is required by Hugging Face pipelines; install inside venv using:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- Manage Python imports correctly. Avoid syntax errors by separating imports and statements.
- Use appropriate `transformers`, `sounddevice`, and auxiliary libraries installed in the working environment.
- Disable GPU if none is available to avoid errors.

***

### 8. Multi-Modal AI Overview

While speech models focus on audio → text tasks, the file also briefly covers:
- Text generation with GPT models.
- Image generation with Stable Diffusion via Hugging Face.
- Text-to-speech synthesis.
- Optical character recognition (OCR).
- Multi-modal chatbots combining text, image, and audio understanding.

This demonstrates the convergence of various generative AI technologies.

***

### 9. Multi-Modality in AI Models (Expanded)

Multi-modal AI models process and integrate multiple input types such as text, images, audio, and video for enhanced understanding and generation.

**Multi-Modal Model Examples and Use Cases:**

1. **Text + Image → Text (Image Captioning / Q&A)**
   - Models: BLIP-2, LLaVA, MiniGPT-4
   - Hugging Face example:
   ```python
   from transformers import BlipProcessor, BlipForConditionalGeneration
   from PIL import Image

   processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
   model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

   image = Image.open("example.jpg")
   inputs = processor(image, return_tensors="pt")
   out = model.generate(**inputs)
   print(processor.decode(out[0], skip_special_tokens=True))
   ```

2. **Text + Text → Image (Text-to-Image Generation)**
   - Models: Stable Diffusion, Kandinsky
   - Usage:
   ```python
   from diffusers import StableDiffusionPipeline
   import torch

   pipe = StableDiffusionPipeline.from_pretrained(
       "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
   ).to("cuda")

   image = pipe("A futuristic city at sunset").images[0]
   image.save("generated.png")
   ```

3. **Image → Text-to-Text (OCR)**
   - Models: TrOCR, Donut
   - Usage:
   ```python
   from transformers import TrOCRProcessor, VisionEncoderDecoderModel
   from PIL import Image

   processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
   model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

   image = Image.open("scanned_text.png")
   pixel_values = processor(image, return_tensors="pt").pixel_values
   generated_ids = model.generate(pixel_values)
   print(processor.batch_decode(generated_ids, skip_special_tokens=True))
   ```

4. **Text + Audio → Text (Speech Recognition)**
   - Model: OpenAI Whisper (via Hugging Face)
   - Usage:
   ```python
   from transformers import pipeline

   asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
   result = asr("sample.wav")
   print(result["text"])
   ```

5. **Text → Audio (Text-to-Speech)**
   - Models: SpeechT5, Bark
   - Usage:
   ```python
   from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
   import soundfile as sf

   processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
   model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
   vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

   inputs = processor(text="Hello, this is a test.", return_tensors="pt")
   speech = model.generate_speech(inputs["input_ids"], vocoder=vocoder)
   sf.write("speech.wav", speech.numpy(), samplerate=16000)
   ```

6. **Audio → Text + Translation**
   - Model: Whisper with `task="translate"`
   - Usage similar to speech recognition.

7. **Video + Text → Text (Video Understanding)**
   - Models: Video-BLIP, InternVideo (under research)

8. **Tri-Modal Models (Text + Image + Audio)**
   - Example: GPT-4o, Kosmos-2, Flamingo (DeepMind)

***

### 10. Unified Multi-Modal AI Program (Python)

A single script to test all multi-modal AI tasks locally:

```python
import torch
from transformers import (
    pipeline,
    BlipProcessor, BlipForConditionalGeneration,
    TrOCRProcessor, VisionEncoderDecoderModel,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
)
from diffusers import StableDiffusionPipeline
from PIL import Image
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def text_to_image():
    prompt = input("Enter text prompt: ")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device=="cuda" else torch.float32
    ).to(device)
    image = pipe(prompt).images[0]
    image.save("generated.png")
    print("✅ Image saved as generated.png")

def image_to_text():
    img_path = input("Enter image path: ")
    image = Image.open(img_path)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    inputs = processor(image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    print("Caption:", caption)

def image_to_ocr():
    img_path = input("Enter image path: ")
    image = Image.open(img_path)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Extracted Text:", text)

def speech_to_text():
    audio_path = input("Enter audio path: ")
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if device=="cuda" else -1)
    result = asr(audio_path)
    print("Transcription:", result["text"])

def speech_to_text_translate():
    audio_path = input("Enter audio path: ")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        generate_kwargs={"task": "translate"},
        device=0 if device=="cuda" else -1
    )
    result = asr(audio_path)
    print("Translated to English:", result["text"])

def text_to_speech():
    text = input("Enter text to speak: ")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    inputs = processor(text=text, return_tensors="pt").to(device)
    speech = model.generate_speech(inputs["input_ids"], vocoder=vocoder)
    sf.write("speech.wav", speech.cpu().numpy(), samplerate=16000)
    print("✅ Audio saved as speech.wav")

def main():
    while True:
        print("\n===== Multi-Modal AI Menu =====")
        print("1. Text → Image (Stable Diffusion)")
        print("2. Image → Text (BLIP-2 Captioning)")
        print("3. Image → Text (OCR TrOCR)")
        print("4. Audio → Text (Whisper ASR)")
        print("5. Audio → English Translation (Whisper)")
        print("6. Text → Speech (SpeechT5 TTS)")
        print("0. Exit")
        choice = input("Choose option: ")

        if choice == "1":
            text_to_image()
        elif choice == "2":
            image_to_text()
        elif choice == "3":
            image_to_ocr()
        elif choice == "4":
            speech_to_text()
        elif choice == "5":
            speech_to_text_translate()
        elif choice == "6":
            text_to_speech()
        elif choice == "0":
            break
        else:
            print("Invalid choice, try again.")

if __name__ == "__main__":
    main()
```

**Setup Instructions:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers diffusers accelerate soundfile pillow
pip install git+https://github.com/openai/whisper.git
```

Run the program:
```bash
python multimodal_ai.py
```

***

### 11. Key Takeaways

- Whisper is a powerful open-source multi-lingual speech recognition and translation model.
- Hugging Face makes Whisper accessible without API keys, usable locally on CPU.
- Multi-modal AI extends capabilities by combining text, image, audio, and video inputs.
- Multi-modal applications enable richer user interactions and new use cases across industries.
- Proper environment setup is key to smooth deployment on Ubuntu and similar platforms.

***
