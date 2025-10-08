import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import logging
import asyncio
import subprocess
import numpy as np
from faster_whisper import WhisperModel
import torch

logging.basicConfig(level=logging.INFO)

app = FastAPI()

if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
    logging.info("GPU detected. Using CUDA with float16 compute type.")
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
    logging.info("No GPU detected. Using CPU with int8 compute type.")

MODEL_SIZE = "large-v3"
logging.info(f"Loading Whisper model: {MODEL_SIZE}")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
logging.info("Whisper model loaded successfully.")

SAMPLING_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE_BYTES = (SAMPLING_RATE * FRAME_DURATION_MS // 1000) * 2
CHUNK_DURATION_SECONDS = 5


async def transcribe_and_send(audio_bytes: bytes, websocket: WebSocket):
    logging.info(f"Transcribing audio chunk of size: {len(audio_bytes)} bytes")
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        segments, _ = model.transcribe(audio_np, beam_size=5)
        transcribed_text = " ".join(segment.text for segment in segments)
        logging.info(f"Transcription result: '{transcribed_text}'")
        if transcribed_text.strip():
            await websocket.send_text(transcribed_text)
    except Exception as e:
        logging.error(f"Error during transcription: {e}")


async def audio_processing_pipeline(websocket: WebSocket):
    audio_queue = asyncio.Queue()

    ffmpeg_command = [
        "ffmpeg", "-i", "-", "-f", "s16le", "-ar", str(SAMPLING_RATE), "-ac", "1", "-"
    ]

    process = await asyncio.create_subprocess_exec(
        *ffmpeg_command,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    try:
        async def feed_ffmpeg(ws: WebSocket):
            while True:
                try:
                    audio_chunk = await ws.receive_bytes()
                    if process.stdin.is_closing():
                        break
                    process.stdin.write(audio_chunk)
                    await process.stdin.drain()
                except WebSocketDisconnect:
                    break
            if not process.stdin.is_closing():
                process.stdin.close()

        async def read_ffmpeg_output():
            while True:
                pcm_chunk = await process.stdout.read(FRAME_SIZE_BYTES)
                if not pcm_chunk:
                    break
                await audio_queue.put(pcm_chunk)
            await audio_queue.put(None)

        async def process_transcription(ws: WebSocket):
            speech_buffer = bytearray()
            buffer_size_limit = int(SAMPLING_RATE * 2 * CHUNK_DURATION_SECONDS)

            while True:
                pcm_chunk = await audio_queue.get()
                if pcm_chunk is None:
                    break

                speech_buffer.extend(pcm_chunk)

                if len(speech_buffer) >= buffer_size_limit:
                    asyncio.create_task(transcribe_and_send(bytes(speech_buffer), ws))
                    speech_buffer.clear()

            if speech_buffer:
                asyncio.create_task(transcribe_and_send(bytes(speech_buffer), ws))

        task1 = asyncio.create_task(feed_ffmpeg(websocket))
        task2 = asyncio.create_task(read_ffmpeg_output())
        task3 = asyncio.create_task(process_transcription(websocket))

        await asyncio.gather(task1, task2, task3)

    except Exception as e:
        logging.error(f"Error during audio processing: {e}")
    finally:
        logging.info("Audio processing finished. Cleaning up FFmpeg.")
        if process.returncode is None:
            process.terminate()
        await process.wait()
        stderr_output = await process.stderr.read()
        if stderr_output:
            logging.error(f"FFmpeg stderr: {stderr_output.decode()}")


@app.get("/")
async def get():
    return FileResponse("templates/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("WebSocket connection established.")
    try:
        await audio_processing_pipeline(websocket)
    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"WebSocket Error: {e}")
    finally:
        logging.info("WebSocket connection closed.")