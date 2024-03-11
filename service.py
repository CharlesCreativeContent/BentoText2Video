from __future__ import annotations

import os
import typing as t
from pathlib import Path
from annotated_types import Ge, Le
from typing_extensions import Annotated

import bentoml
#SETUP for (vllm) text generation api
import uuid
from typing import AsyncGenerator
import asyncio
#SETUP for (sdxl-turbo) image generation api
from PIL.Image import Image
#SETUP for (moviepy) video builder api
from moviepy.editor import *

#CONSTANTS for (xtts) audio generation api
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
sample_input_data = {
    'text': 'It took me quite a long time to develop a voice and now that I have it I am not going to be silent.',
    'language': 'en',
}
#CONSTANTS for (sdxl-turbo) image generation api
MODEL_ID = "stabilityai/sdxl-turbo"
sample_prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

#CONSTANTS for (vllm) text generation api
MAX_TOKENS = 256
VLLM_Model = 'meta-llama/Llama-2-7b-chat-hf'
sample_vllm_prompt = "Happy New Year BentoML team, Wishing you "
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for creating videos. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{user_prompt} [/INST] """


@bentoml.service(
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
        "memory": "8Gi",
    },
)
class VLLM:

    @bentoml.api
    def __init__(self) -> None:
        from vllm import LLM, SamplingParams
        self.sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_token=MAX_TOKENS)
        self.llm = LLM(model=VLLM_Model)

    def generate(
        self,
        prompt: str = sample_vllm_prompt,
    ) -> str:

        outputs = self.llm.generate([prompt], self.sampling_params)
        # Print the outputs.
        generation = ""
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            generation += generated_text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return generation

@bentoml.service(
    resources={
        "gpu": 2,
        "gpu_type": "nvidia-l4",
        "memory": "8Gi",
    },
    traffic={"timeout": 300},
)

class XTTS:
    def __init__(self) -> None:
        import torch
        from TTS.api import TTS

        self.tts = TTS(TTS_MODEL, gpu=torch.cuda.is_available())
    
    @bentoml.api
    def synthesize(
            self,
            context: bentoml.Context,
            script: str = sample_input_data["text"],
            lang: str = sample_input_data["language"],
    ) -> t.Annotated[Path, bentoml.validators.ContentType('audio/*')]:
        output_path = os.path.join(context.temp_dir, "output.wav")
        sample_path = "./female.wav"
        if not os.path.exists(sample_path):
            sample_path = "./src/female.wav"

        self.tts.tts_to_file(
            text=script,
            file_path=output_path,
            speaker_wav=sample_path,
            language=lang,
            split_sentences=True,
        )
        return Path(output_path)


@bentoml.service(
traffic={"timeout": 300},
workers=1,
resources={
    "gpu": 1,
    "gpu_type": "nvidia-l4",
},
)
class SDXLTurbo:
    def __init__(self) -> None:
        from diffusers import AutoPipelineForText2Image
        import torch

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to(device="cuda")

    @bentoml.api
    def txt2img(
            self,
            prompt: str = sample_prompt,
            num_inference_steps: Annotated[int, Ge(1), Le(10)] = 1,
            guidance_scale: float = 0.0,
    ) -> Image:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        return image


@bentoml.service(
traffic={"timeout": 300},
workers=1,
resources={
    "gpu": 1,
    "gpu_type": "nvidia-l4",
},
)

class Text2Video:
    vllm_service = bentoml.depends(VLLM)
    sdxl_service = bentoml.depends(SDXLTurbo)
    xtts_service = bentoml.depends(XTTS)
    
    @bentoml.api
    def txt2video(
            self,
            context: bentoml.Context,
            text: str = sample_vllm_prompt,
            lang: str = sample_input_data["language"],
    )  -> t.Annotated[Path, bentoml.validators.ContentType("video/*")]:
        #Generate Text Script
        generatedText = self.vllm_service.generate(prompt=text)
        script = text + generatedText
        #Generate Image
        image = self.sdxl_service.txt2img(prompt=text, num_inference_steps = 1,guidance_scale=0)
        imageFilename  = "outputImage.jpg"
        image.save(imageFilename)
        #Generate Audio Clip
        audioPath = self.xtts_service.synthesize(context=context,script=script,lang=lang)
        audioFilePath = str(audioPath)
        audio = AudioFileClip(audioFilePath)
        #Edit Video Together
        clip = ImageClip(imageFilename).set_duration(audio.duration)
        clip = clip.set_audio(audio)
        output_path = os.path.join(context.temp_dir, "outputVideo.mp4")
        clip.write_videofile(output_path, fps=24)
        
        return Path(output_path)
