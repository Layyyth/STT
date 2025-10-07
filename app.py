from fastapi import FastAPI , WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import logging
import asyncio

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# vad settings
SAMPLING_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE_BYTES = ( SAMPLING_RATE * FRAME_DURATION_MS // 1000 ) * 2

async def audio_processing_pipeline(websocket: WebSocket):
    # FFMPEG decode and reformat
    ffmpeg_command = [
        "ffmpeg",
        "-i","-",
        "-f","s16le",
        "-ar",str(SAMPLING_RATE),
        "-ac","1",
        "-"
    ]

    logging.info("Starting ffmpeg process")
    process = await asyncio.create_subprocess_exec(
        *ffmpeg_command,
        stdin= subprocess.PIPE,
        stdout= subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    try:
        # this pile of shit to read from websocket and feed ffmpeg stdin and read from ffmpeg stdout and process audio with vad
        async def feed_ffmpeg(ws:WebSocket):
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

        # raw audio after decoding is read and apply vad
        async def process_vad():
            while True:
                pcm_chunk = await process.stdout.read(FRAME_SIZE_BYTES)
                if not pcm_chunk:
                    break
                logging.info(f"Received {len(pcm_chunk)} bytes of PCM audio from ffmpeg")

        Task1 = asyncio.create_task(feed_ffmpeg(websocket))
        Task2 = asyncio.create_task(process_vad())
        await asyncio.gather(Task1, Task2)

    except Exception as e:
        logging.error(f"Error during audio processing: {e}")
    finally :
        logging.info("Audio Processing Finished. Cleaning up FFmpeg")
        if process.returncode is None :
            process.terminate()
        await process.wait()

        stderr_output = await process.stderr.read()
        if  stderr_output :
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



