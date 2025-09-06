# Cursor

*Audio-Driven Sampling for LLMs*

This project is a prototype exploring how to guide Large Language Model (LLM) inference by modulating the sampling temperature in real time using the loudness of a music track.

## The Idea
Music encodes stochastic patterns similar to those shaping human cognition. By linking this musical variation to the temperature parameter, the model’s latent space can be explored under a human-like rhythm. In practice, the loudness profile of a song becomes a stand-in for the probability distribution that a brain might use when sampling patterns. This lets the LLM follow the “frequency” of the music while generating text. Currently, this setup enables testing the influence of different genres—jazz, techno, silence as a baseline—on text generation when tied to temperature.

## How It Works
The system processes an audio track and uses its loudness contour to modulate the LLM’s temperature parameter during token-by-token text creation.
1. Audio Processing: The input track is analyzed frame by frame (LUFS) in sync with the target tokens-per-second rate.  
2. Dynamic Intensity: Loudness values are smoothed, then differentiated to capture their rate of change (“loudness velocity”).  
3. Temperature Control: This velocity is mapped to a temperature range. Calm segments lower the temperature, dynamic peaks raise it.  
4. Token Generation: The script communicates with a local LLM server, feeding both conversation history and the calculated temperature.  
5. Interactive Playback: A TUI shows the output text as it’s generated, aligned with the music playback, with options to pause, seek, or restart.  
6. Session Storage: Prompts, audio analysis, and generated tokens can be saved to disk for replay later.  

## Setup
1. Install Python packages: pip install -r requirements.txt  
   (On some systems, PyAudio may need manual installation via your package manager.)  
2. Install system libraries for audio playback (if required):  
   Debian/Ubuntu: sudo apt-get install libportaudio2  
   Fedora: sudo dnf install portaudio  
   Arch Linux: sudo pacman -S portaudio  
3. Run a local LLM server: Works with an OpenAI-compatible local server such as LM Studio (https://lmstudio.ai/).  
   - Install LM Studio.  
   - Download a model from the search tab.  
   - Start the server on localhost:1234.  

## Usage
Generate with audio modulation:  
```bash
python main.py
```
Custom song and prompt:  
```bash
python main.py --song-path "path/to/song.mp3" --prompt "Why is the sky blue?"  
```
The script will play the song and display generated text in sync with the music.  

Replay a saved session:  
```bash
python main.py --load-run .runs/run_YYYY-MM-DD_HH-MM-SS.npz  
```

## The Road Ahead
This is just a proof of concept. The building blocks can be recombined in many ways: routing energy, shaping attractor dynamics, or coupling stochastic processes. 
One vision is to pair this with models like Magenta RT, letting an LLM generate prompts for a music generator in parallel with its main dialogue. 
The music then acts as a stochastic side-channel shaping the LLM’s reasoning loop. Another path (pursued at frontier labs) is to let the LLM itself output a continuous-valued temperature per token, training it via reinforcement learning. 
This embeds a micro-reasoning layer inside the model rather than relying on an external stochastic driver. We’re only beginning to explore the possibilities.

