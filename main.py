# Audio-driven token generation demo for LLMs (temperature follows music dynamics)
# ------------------------------------------------------------------------------
# This program ties the sampling temperature of a local, OpenAI-compatible LLM
# to features extracted from an audio track (e.g., loudness and its velocity).
# It provides a Rich-based TUI that streams colored tokens in sync with playback,
# supports seeking/pausing, and can save+replay complete runs.

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
import logging
import queue
import threading
import io
import pickle
import sys
import tty
import termios
import select

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import audio
from just_playback import Playback
from openai import OpenAI, Timeout, APITimeoutError
import maths
from maths import scurve as sc, jcurve as jc, rcurve as rc, clamp01

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
from rich.box import HEAVY
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn


@dataclass
class TimedToken:
    token: str
    timestamp: float  # seconds since start
    loudness: float   # normalized in [0, 1]


@dataclass
class PlaybackState:
    playing: bool = True
    restart: bool = False
    quit: bool = False
    seek_by: int = 0           # token offset
    seek_to_start: bool = False
    toggle_pause_request: bool = False


@dataclass
class AppConfig:
    """
    App settings for audio-reactive generation.
    """
    song_path: Path
    prompt: str
    lm_studio_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    tps: int = 45              # tokens per second
    min_temp: float = 0.985
    max_temp: float = 1.875
    max_tokens: int = 50
    save_run_path: Path | None = None
    _is_loaded_run: bool = False  # prevents auto save path when loading

    def __post_init__(self):
        if self.save_run_path is None and not self._is_loaded_run:
            runs_dir = Path(".runs")
            runs_dir.mkdir(exist_ok=True)
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.save_run_path = runs_dir / f"run_{ts}.npz"


@dataclass
class GenerationRun:
    """
    Snapshot of a session for later replay.
    """
    config: AppConfig
    timed_tokens: list[TimedToken]
    temp: np.ndarray
    lufs: np.ndarray
    lufs_raw: np.ndarray
    v_lufs: np.ndarray

    def save(self, path: Path):
        """Write run bundle to compressed .npz."""
        tokens = [t.token for t in self.timed_tokens]
        timestamps = [t.timestamp for t in self.timed_tokens]
        loudness_values = [t.loudness for t in self.timed_tokens]

        cfg = {
            'song_path': str(self.config.song_path),
            'prompt': self.config.prompt,
            'lm_studio_url': self.config.lm_studio_url,
            'model': self.config.model,
            'tps': self.config.tps,
            'min_temp': self.config.min_temp,
            'max_temp': self.config.max_temp,
            'max_tokens': self.config.max_tokens,
            'save_run_path': str(self.config.save_run_path) if self.config.save_run_path else 'None'
        }

        np.savez_compressed(
            path,
            **cfg,
            tokens=np.array(tokens, dtype=object),
            timestamps=np.array(timestamps),
            loudness_values=np.array(loudness_values),
            temp=self.temp,
            lufs=self.lufs,
            lufs_raw=self.lufs_raw,
            v_lufs=self.v_lufs
        )

    @classmethod
    def load(cls, path: Path) -> 'GenerationRun':
        """Restore a run from .npz."""
        with np.load(path, allow_pickle=True) as data:
            cfg = AppConfig(
                song_path=Path(str(data['song_path'])),
                prompt=str(data['prompt']),
                lm_studio_url=str(data['lm_studio_url']),
                model=str(data['model']),
                tps=int(data['tps']),
                min_temp=float(data['min_temp']),
                max_temp=float(data['max_temp']),
                max_tokens=int(data['max_tokens']),
                save_run_path=Path(str(data['save_run_path'])) if str(data['save_run_path']) != 'None' else None,
                _is_loaded_run=True
            )
            timed_tokens = [
                TimedToken(token=t, timestamp=ts, loudness=l)
                for t, ts, l in zip(data['tokens'], data['timestamps'], data['loudness_values'])
            ]
            return GenerationRun(
                config=cfg,
                timed_tokens=timed_tokens,
                temp=data['temp'],
                lufs=data['lufs'],
                lufs_raw=data['lufs_raw'],
                v_lufs=data['v_lufs']
            )


def colorize_token(token: str, loudness: float) -> str:
    """
    Map token color to loudness via Viridis colormap on black background.
    """
    cmap = cm.get_cmap('viridis')
    gamma = 0.75  # brighten darker tones
    r_f, g_f, b_f = cmap(loudness)[:3]
    r, g, b = [int((c ** gamma) * 255) for c in (r_f, g_f, b_f)]
    fg = f"\x1b[38;2;{r};{g};{b}m"
    bg = "\x1b[48;2;0;0;0m"
    reset = "\x1b[0m"
    return f"{bg}{fg}{token}{reset}"


class GenerationTimeoutError(Exception):
    pass


def get_client(config: AppConfig) -> OpenAI:
    """
    Build client for local OpenAI-compatible endpoint (e.g., LM Studio).
    """
    return OpenAI(base_url=config.lm_studio_url, api_key="not-needed", timeout=Timeout(5.0))


def get_token_worker(client, config, prompt, temperature, result_queue):
    """
    Fire a single-token streaming request in a background thread and push the first chunk or error.
    """
    try:
        response = client.completions.create(
            model=config.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=1,
            stream=True,
        )
        chunk = next(response)
        result_queue.put(chunk)
    except Exception as e:
        result_queue.put(e)


def generate_all_tokens(
    client: OpenAI,
    config: AppConfig,
    temperature: np.ndarray,
    live: Live,
    header_text: Text,
    generation_text: Text,
    dynamic_info_text: Text
) -> list[TimedToken]:
    """
    Token-by-token generation loop with per-token temperature modulation.
    """
    prompt_prefix = f"User: {config.prompt}\nAssistant:"
    history = ""
    timed_tokens: list[TimedToken] = []
    token_index = 0
    MAX_HISTORY_CHARS = 2048

    header_text.plain = "Generating tokens..."
    live.refresh()

    while token_index < config.max_tokens:
        frame_index = min(token_index, len(temperature) - 1)
        temp = temperature[frame_index]
        now_s = token_index / config.tps

        dynamic_info_text.plain = (
            f"Token: {token_index + 1} / {config.max_tokens}\n"
            f"Time: {now_s:.2f}s\n"
            f"Temperature: {temp:.3f}"
        )

        loudness_value = (temp - config.min_temp) / (config.max_temp - config.min_temp)
        prompt = prompt_prefix + history

        try:
            result_queue = queue.Queue()
            worker = threading.Thread(
                target=get_token_worker,
                args=(client, config, prompt, temp, result_queue),
                daemon=True
            )
            worker.start()

            # Fail-safe for stalled calls
            result = result_queue.get(timeout=5.0)
            if isinstance(result, Exception):
                raise result

            chunk = result
            token = chunk.choices[0].text
            finish_reason = chunk.choices[0].finish_reason

            colorized = colorize_token(token.replace('\n', ' '), loudness_value)
            generation_text.append(Text.from_ansi(colorized))
            live.refresh()

            timestamp = token_index / config.tps
            timed_tokens.append(TimedToken(token, timestamp, loudness_value))

            history += token
            if len(history) > MAX_HISTORY_CHARS:
                history = history[-MAX_HISTORY_CHARS:]

            token_index += 1
            if finish_reason == "stop":
                break

        except queue.Empty:
            logging.warning("Token worker unresponsive; aborting generation.")
            break
        except Exception as e:
            logging.error(f"Generation error: {e}")
            break

    header_text.plain = "Generation complete."
    live.refresh()
    return timed_tokens


def run_tui(
    config: AppConfig,
    temp: np.ndarray,
    lufs: np.ndarray,
    lufs_raw: np.ndarray,
    v_lufs: np.ndarray,
    timed_tokens: list[TimedToken] | None = None,
    is_loaded_run: bool = False
) -> list[TimedToken] | None:
    """
    Interactive TUI for generation and synchronized playback.
    """
    console = Console()
    log_stream = io.StringIO()
    log_console = Console(file=log_stream, force_terminal=True, width=120)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt="[%X]",
        handlers=[RichHandler(console=log_console, rich_tracebacks=True, show_path=False)],
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    layout = Layout()
    layout.split(
        Layout(name="header", size=5),
        Layout(name="main"),
        Layout(size=10, name="footer"),
    )
    layout["main"].split_row(
        Layout(name="main_content"),
        Layout(name="side_panel", size=40, minimum_size=35)
    )

    header_text = Text(f"Prompt: {config.prompt}\nSong: {config.song_path}", justify="center")
    generation_text = Text("", justify="left")
    playback_text = Text("", justify="left")

    layout["main_content"].update(Panel(generation_text, title="Generation", border_style="green", box=HEAVY))

    log_panel = Panel("", title="[bold cyan]Logs[/]", border_style="red", box=HEAVY)
    layout["footer"].update(log_panel)

    song_duration = lufs_raw.shape[0] / config.tps
    lufs_raw_stats = f"{np.min(lufs_raw):.2f} -> {np.max(lufs_raw):.2f} (μ: {np.mean(lufs_raw):.2f})"
    lufs_proc_stats = f"{np.min(lufs):.2f} -> {np.max(lufs):.2f} (μ: {np.mean(lufs):.2f})"
    v_lufs_stats = f"{np.min(v_lufs):.2f} -> {np.max(v_lufs):.2f} (μ: {np.mean(v_lufs):.2f})"
    temp_stats = f"{np.min(temp):.2f} -> {np.max(temp):.2f} (μ: {np.mean(temp):.2f})"

    info_table = Table(show_header=False, box=None, padding=(0, 1))
    info_table.add_column(style="bold blue")
    info_table.add_column(style="white")
    info_table.add_row("Song Duration:", f"{song_duration:.2f}s")
    info_table.add_row("Tokens to Gen:", str(config.max_tokens))
    info_table.add_row("LUFS (raw):", lufs_raw_stats)
    info_table.add_row("LUFS (proc):", lufs_proc_stats)
    info_table.add_row("Velocity:", v_lufs_stats)
    info_table.add_row("Temperature:", temp_stats)

    dynamic_info_text = Text("", justify="left")
    info_group = Group(
        Panel(info_table, title="[bold yellow]Audio Analysis[/]", border_style="yellow", box=HEAVY),
        Panel(dynamic_info_text, title="[bold yellow]Live Stats[/]", border_style="yellow", box=HEAVY)
    )
    layout["side_panel"].update(info_group)

    layout["header"].update(
        Panel(header_text, title="[bold magenta]DJ-Token[/]", border_style="magenta", box=HEAVY)
    )

    final_console = Console()
    tokens = timed_tokens

    # Non-blocking stdin
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(layout, console=console, screen=True, refresh_per_second=60, redirect_stderr=False, transient=True) as live:
            def update_logs():
                log_panel.renderable = Text.from_ansi(log_stream.getvalue())
                live.refresh()

            update_logs()

            # Generate if needed
            if not tokens:
                client = get_client(config)
                tokens = generate_all_tokens(client, config, temp, live, header_text, generation_text, dynamic_info_text)
                update_logs()
            if not tokens:
                logging.error("No tokens produced; exiting.")
                update_logs()
                return

            # Playback & controls
            playstate = PlaybackState()
            header_text.plain = f"Playing '{config.song_path}'"
            layout["main_content"].update(Panel(playback_text, title="Playback", border_style="blue", box=HEAVY))

            pbar = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            )
            pbar_task = pbar.add_task("Loading...", total=len(tokens))

            cmd_text = "P/Space: Play/Pause | H: Home | ←/→: Seek | Q: Quit"
            if not is_loaded_run:
                cmd_text = "R: Restart | " + cmd_text
            command_bar = Panel(Group(pbar, Text(cmd_text, justify="center")), title="[bold]Commands[/]", border_style="green", box=HEAVY)
            layout["footer"].split(Layout(log_panel), Layout(command_bar, name="command_bar", size=5))
            live.refresh()

            playback = Playback()
            playback.load_file(str(config.song_path))
            playback.play()

            FPS = 60.0
            song_duration = lufs_raw.shape[0] / config.tps
            inext = 0
            t = 0.0

            def redraw_playback_text(idx: int):
                playback_text.plain = ""
                for i in range(idx):
                    tok = tokens[i]
                    clean = tok.token.strip()
                    if clean:
                        playback_text.append(Text.from_ansi(colorize_token(clean, tok.loudness)))
                        if not (clean.endswith('.') or clean.endswith('!') or clean.endswith('?')):
                            playback_text.append(" ", style="white on black")
                live.refresh()

            while not playstate.quit:
                # keyboard
                if select.select([sys.stdin], [], [], 1.0 / FPS) == ([sys.stdin], [], []):
                    keys = ''
                    while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        keys += sys.stdin.read(1)
                    if keys:
                        logging.info(f"Key: {keys!r}")
                        update_logs()

                    if keys in ('p', ' '):
                        playstate.playing = not playstate.playing
                        playstate.toggle_pause_request = True
                    elif keys == 'r' and not is_loaded_run:
                        playstate.restart = True
                    elif keys == 'h':
                        playstate.seek_to_start = True
                    elif keys == 'q':
                        playstate.quit = True
                    elif keys == '\x1b[D':
                        playstate.seek_by = -20
                    elif keys == '\x1b[C':
                        playstate.seek_by = 20

                if playstate.restart:
                    playback.stop()
                    raise Exception("Restarting")

                if playstate.toggle_pause_request:
                    if playback.paused:
                        playback.resume()
                    else:
                        playback.pause()
                    playstate.toggle_pause_request = False

                # seeking
                if playstate.seek_by != 0:
                    inew = max(0, min(inext + playstate.seek_by, len(tokens) - 1))
                    if inew != inext:
                        inext = inew
                        t = tokens[inext].timestamp
                        playback.seek(t)
                        redraw_playback_text(inext)
                    playstate.seek_by = 0

                if playstate.seek_to_start:
                    inext = 0
                    t = 0.0
                    playback.seek(t)
                    redraw_playback_text(inext)
                    playstate.seek_to_start = False

                if playstate.playing:
                    pbar.update(pbar_task, description="Playing")
                    catchup_rate = 0.0
                    if playstate.seek_by == 0:
                        t_playback = playback.curr_pos
                        f = min(int(t_playback * config.tps), lufs.shape[0] - 1)

                        K_MIN, K_MAX = 0.05, 8.0
                        dt = 1.0 / FPS
                        lag = t_playback - t
                        lag_t = float(clamp01(lag / 0.5))
                        catchup_rate = K_MIN + (K_MAX - K_MIN) * sc(v_lufs[f], lag_t * 0.5)
                        t += dt * catchup_rate
                        t = float(maths.clamp(t, 0, t_playback))

                    if inext >= len(tokens):
                        inext, t = 0, 0.0
                        playback_text.plain = ""
                        playback.play()
                        live.refresh()
                        continue

                    t_playback = playback.curr_pos
                    f = min(int(t_playback * config.tps), lufs.shape[0] - 1)
                    dynamic_info_text.plain = (
                        f"Audio Time: {t_playback:.2f}s / {song_duration:.2f}s\n"
                        f"Virtual Time: {t:.2f}s\n"
                        f"Tokens: {inext} / {len(tokens)}\n"
                        f"Loudness: {lufs[f]:.3f}\n"
                        f"Catch-up rate: {catchup_rate:.2f}x"
                    )
                    pbar.update(pbar_task, completed=inext)

                    advanced = False
                    while inext < len(tokens) and tokens[inext].timestamp <= t:
                        obj = tokens[inext]
                        clean = obj.token.strip()
                        if clean:
                            playback_text.append(Text.from_ansi(colorize_token(clean, obj.loudness)))
                            if not (clean.endswith('.') or clean.endswith('!') or clean.endswith('?')):
                                playback_text.append(" ", style="white on black")
                        inext += 1
                        advanced = True
                    if advanced:
                        live.refresh()

                else:
                    if playback.active:
                        playback.pause()
                    dynamic_info_text.plain = "[bold red]PAUSED[/]\n\n" + dynamic_info_text.plain.split('\n\n')[-1]
                    pbar.update(pbar_task, description="Paused")
                    live.refresh()

            if playback.active:
                playback.stop()
            header_text.plain = "Playback finished."
            dynamic_info_text.plain = "Done."
            live.refresh()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    final_console.print(Panel(Text(config.prompt, justify="center"), title="Prompt", border_style="green"))
    final_text = Text()
    for tt in tokens:
        final_text.append(tt.token)
    final_console.print(Panel(final_text, title="Final Generated Text", border_style="blue"))
    return tokens


def main(config: AppConfig):
    """
    End-to-end: analyze audio, preview curves, then interactive generation/playback.
    """
    console = Console()
    log_stream = io.StringIO()
    log_console = Console(file=log_stream, force_terminal=True, width=120)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        datefmt="[%X]",
        handlers=[RichHandler(console=log_console, rich_tracebacks=True, show_path=False)],
        force=True
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("Loading and processing audio features...")
    maths.fps = config.tps
    lufs_raw = audio.load_lufs(config.song_path, fps=config.tps)
    maths.n = lufs_raw.shape[0]

    if lufs_raw.shape[0] == 0:
        logging.error("No loudness data found (file missing or silent).")
        console.print(Text.from_ansi(log_stream.getvalue()))
        return

    lufs = maths.scurve(maths.wavg(lufs_raw, window=0.1), 0.75, 3.0)
    v_lufs = maths.absdiff(lufs)
    v_lufs = v_lufs / np.max(v_lufs) if np.max(v_lufs) > 0 else np.zeros_like(v_lufs)
    temp = config.min_temp + (config.max_temp - config.min_temp) * v_lufs

    logging.info("Audio analysis complete.")
    logging.info("Close the plot window to proceed to the TUI.")
    console.print(Panel(Text.from_ansi(log_stream.getvalue()), title="[bold cyan]Startup[/]", border_style="cyan"))

    # Visualize analysis
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    frame_axis = np.arange(lufs_raw.shape[0])

    ax1.plot(frame_axis, lufs_raw)
    ax1.set_title("Original Loudness")
    ax1.set_ylabel("Norm Loudness (0-1)")

    ax2.plot(frame_axis, lufs)
    ax2.set_title("Processed Loudness (Smoothed + S-Curve)")
    ax2.set_ylabel("Processed (0-1)")

    ax3.plot(frame_axis, v_lufs)
    ax3.set_title("Loudness Velocity")
    ax3.set_ylabel("Velocity (0-1)")

    ax4.plot(frame_axis, temp)
    ax4.set_title("Temperature Over Time")
    ax4.set_xlabel("Frames (≈ tokens)")
    ax4.set_ylabel("Temperature")

    plt.tight_layout()
    plt.show()

    # Launch TUI
    timed_tokens = run_tui(config, temp, lufs, lufs_raw, v_lufs)

    if config.save_run_path and timed_tokens:
        logging.info(f"Saving run to {config.save_run_path}...")
        GenerationRun(
            config=config,
            timed_tokens=timed_tokens,
            temp=temp,
            lufs=lufs,
            lufs_raw=lufs_raw,
            v_lufs=v_lufs
        ).save(config.save_run_path)
        logging.info("Run saved.")


def app_main():
    while True:
        try:
            parser = argparse.ArgumentParser(description="Audio-reactive token generation with LLMs.")
            parser.add_argument("--song-path", type=Path, default="inputs/miles-pharaoh.flac", help="Path to audio file.")
            parser.add_argument("--prompt", default="Describe a beautiful sunset in ~600 words.", help="LLM prompt.")
            parser.add_argument("--max-tokens", type=int, default=50, help="Upper limit of tokens to sample.")
            parser.add_argument("--load-run", type=Path, help="Load a saved run (.npz) and play it back.")
            parser.add_argument("--save-run", type=Path, help="Override default save path (.runs/...)")

            args = parser.parse_args()

            if args.load_run:
                try:
                    run_data = GenerationRun.load(args.load_run)
                    Console().print(f"Loaded run from [bold cyan]{args.load_run}[/bold cyan]")
                    run_tui(
                        config=run_data.config,
                        temp=run_data.temp,
                        lufs=run_data.lufs,
                        lufs_raw=run_data.lufs_raw,
                        v_lufs=run_data.v_lufs,
                        timed_tokens=run_data.timed_tokens,
                        is_loaded_run=True
                    )
                except FileNotFoundError:
                    Console().print(f"[bold red]Run file not found: {args.load_run}[/bold red]")
                except Exception as e:
                    Console().print(f"[bold red]Failed to load run: {e}[/bold red]")
                break

            config = AppConfig(
                song_path=args.song_path,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                save_run_path=args.save_run
            )
            main(config)

        except Exception as e:
            if "Restarting" in str(e):
                continue
            Console().print_exception()
            break
        break


if __name__ == "__main__":
    app_main()
# ```
