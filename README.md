<div align="center">
  <h1>🎤 ComfyUI-SoulX-Singer</h1>
  <p>
    ComfyUI custom nodes for<br>
    <b><em>SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis</em></b>
  </p>
  <p>
    <a href="https://soul-ailab.github.io/soulx-singer/"><img src="https://img.shields.io/badge/Demo-Page-lightgrey" alt="Demo Page"></a>
    <a href="https://huggingface.co/spaces/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-Online%20Demo-ffda16" alt="HF Space Demo"></a>
    <a href="https://huggingface.co/drbaph/SoulX-Singer"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue' alt="HF Model"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/GitHub-Original-green" alt="GitHub"></a>
    <a href="https://huggingface.co/papers/2602.07803"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Paper-yellow" alt="HF Paper"></a>
    <a href="https://arxiv.org/abs/2602.07803"><img src="https://img.shields.io/badge/arXiv-2602.07803-b31b1b" alt="arXiv"></a>
    <a href="https://github.com/Soul-AILab/SoulX-Singer"><img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License"></a>
  </p>
</div>

---

## 🎵 Overview

**SoulX-Singer** is a high-fidelity, zero-shot singing voice synthesis model by [SoulAI-Lab](https://github.com/Soul-AILab) that enables users to generate realistic singing voices for unseen singers.  
This ComfyUI wrapper provides native node-based integration with support for **melody-conditioned (F0 contour)** and **score-conditioned (MIDI notes)** control for precise pitch, rhythm, and expression.

**Paper:** [SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis](https://arxiv.org/abs/2602.07803) (arXiv:2602.07803)

---

## ✨ Features

- **🎤 Zero-Shot Singing** – Generate voices for unseen singers with just a reference sample
- **🎵 Dual Control Modes** – Melody (F0 contour) and Score (MIDI notes) conditioning
- **🔗 Native ComfyUI Integration** – AUDIO noodle inputs, progress bars, interruption support
- **⚡ Optimized Performance** – Support for bf16/fp32 dtypes, SDPA and SageAttention
- **📦 Smart Auto-Download** – Downloads only what you need from HuggingFace
  - bf16 model + preprocessors by default (~6GB)
  - Optional fp32 model for maximum quality (~10GB total)
- **💾 Smart Caching** – Optional model caching with dtype/attention change detection
- **🎹 MIDI Editor Support** – Advanced node for manual metadata editing workflow

---

## 📦 Installation

<details>
<summary><b>📥 Click to expand installation methods</b></summary>

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "SoulX-Singer"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/Saganaki22/ComfyUI-SoulX-Singer.git
cd ComfyUI-SoulX-Singer
pip install -r requirements.txt
```

**Note:** The `--recursive` flag is important to clone the SoulX-Singer submodule.

### Method 3: If Already Cloned Without Submodule

```bash
cd ComfyUI/custom_nodes/ComfyUI-SoulX-Singer
git submodule init
git submodule update
pip install -r requirements.txt
```

</details>

---

## 🚀 Quick Start

### Basic Workflow (Simple Mode)

1. **Load Model**
   - Add `🎤 SoulX-Singer Model Loader` node
   - Select model:
     - `SoulX-Singer_model_bf16` (default) - Fast, good quality, ~2GB
     - `SoulX-Singer_model_fp32` - Best quality, ~4GB
     - "(download)" suffix means it will be auto-downloaded on first use
   - Choose dtype: `bf16` (default, recommended) or `fp32` (full precision)
   - Choose attention: `sdpa` (default) or `sageattention` (fastest with sageattention package)
   - Enable `keep_loaded` to cache model between runs

2. **Load Audio**
   - Add `Load Audio` nodes for prompt and target audio
   - Prompt: 3-10 seconds of reference singing voice
   - Target: Audio with melody/score to synthesize

3. **Synthesize**
   - Add `🎙️ SoulX-Singer Simple` node
   - Connect model and audio inputs
   - Configure languages (Mandarin/English/Cantonese)
   - Set control mode (`melody` or `score`)
   - Adjust synthesis parameters
   - Run!

4. **Save/Preview**
   - Connect to `Save Audio` or `Preview Audio` node

### Advanced Workflow (Metadata Mode)

For users who want manual control with MIDI editor:

1. Run Simple mode once to generate metadata files (saved in temp folder)
2. Copy metadata JSON files from temp folder
3. Edit metadata JSON files with [MIDI Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor)
4. Use `🎙️ SoulX-Singer Advanced` node with:
   - Prompt audio file
   - Prompt metadata JSON path
   - Target metadata JSON path (edited version)

**Why no target_audio in Advanced node?** The target is defined entirely by metadata (lyrics, notes, timing) - the node synthesizes NEW audio from scratch rather than transforming existing audio.

---

## 🗂️ File Structure & Downloads

<details>
<summary><b>📁 Click to expand file structure details</b></summary>

### Automatic Download (Recommended)

On first use, the node will automatically download required files from [drbaph/SoulX-Singer](https://huggingface.co/drbaph/SoulX-Singer):

**Default Download (bf16):**
- `SoulX-Singer_model_bf16.safetensors` (~1.5GB)
- `config.yaml`
- `preprocessors/` folder (~5GB)
- **Total:** ~6.5GB

**Optional Download (fp32):**
- `SoulX-Singer_model_fp32.safetensors` (~2.9GB)
- Plus bf16 model + config + preprocessors above
- **Total:** ~9.5GB

Files are saved to:
```
ComfyUI/models/SoulX-Singer/
```

### Manual Download

If auto-download fails:

```bash
pip install -U huggingface_hub
huggingface-cli download drbaph/SoulX-Singer --local-dir ComfyUI/models/SoulX-Singer
```

Or download manually from [drbaph/SoulX-Singer](https://huggingface.co/drbaph/SoulX-Singer) and place in `ComfyUI/models/SoulX-Singer/`.

### Final Structure

```
ComfyUI/
├── models/
│   └── SoulX-Singer/
│       ├── SoulX-Singer_model_bf16.safetensors   # bf16 model (~1.5GB)
│       ├── SoulX-Singer_model_fp32.safetensors   # fp32 model (~2.9GB) [optional]
│       ├── config.yaml                            # Model config
│       └── preprocessors/                         # Preprocessing models (~5GB)
│           ├── dereverb_mel_band_roformer/
│           ├── mel-band-roformer-karaoke/
│           ├── parakeet-tdt-0.6b-v2/
│           ├── rmvpe/
│           ├── rosvot/
│           └── speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
└── custom_nodes/
    └── ComfyUI-SoulX-Singer/
        ├── __init__.py
        ├── nodes/
        │   ├── model_loader.py
        │   ├── simple_synthesizer.py
        │   └── advanced_synthesizer.py
        ├── SoulX-Singer/                       # Git submodule
        ├── requirements.txt
        └── README.md
```

### 🔗 Symlink Support

✅ **All nodes support symlinks!** You can use system links to save disk space:

**Windows Example:**
```cmd
:: Link the entire models directory
mklink /D "ComfyUI\models\SoulX-Singer" "D:\MyModels\SoulX-Singer"

:: Or link just the preprocessors
mklink /D "ComfyUI\models\SoulX-Singer\preprocessors" "D:\MyModels\preprocessors"
```

**Linux/Mac Example:**
```bash
# Link the entire models directory
ln -s /path/to/your/models/SoulX-Singer ComfyUI/models/SoulX-Singer

# Or link just the preprocessors
ln -s /path/to/preprocessors ComfyUI/models/SoulX-Singer/preprocessors
```

The nodes automatically resolve symlinks and load from the actual file location.

</details>

---

## 🎛️ Node Reference

### 🎤 SoulX-Singer Model Loader

Loads the SVS model with configurable precision and attention.

**Inputs:**
- `model_name`: Model file to load
  - `SoulX-Singer_model_bf16` (default) - bf16 precision, fast, good quality
  - `SoulX-Singer_model_fp32` - fp32 precision, best quality, larger file
  - `(download)` suffix appears if model not yet downloaded
  - Automatically detects all `.safetensors` and `.pt` files in `ComfyUI/models/SoulX-Singer/`
  - **Supports symlinks:** Works with symlinked files/directories
- `dtype`: Precision - `bf16` (default, recommended), `fp32` (full)
  - **Note:** fp16 removed due to vocoder FFT incompatibility
- `attention_type`: `sdpa` (default) or `sageattention`
  - **Note:** `auto`, `flash_attention`, and `eager` removed due to compatibility issues
  - `sageattention` requires: `pip install sageattention`
- `keep_loaded`: Cache model in memory (clears on dtype/attention change)

**Outputs:**
- `model`: SOULX_MODEL object

**Smart Download Behavior:**
- Selecting bf16 model: Downloads bf16 + config + preprocessors (~7GB)
- Selecting fp32 model: Downloads fp32 + bf16 + config + preprocessors (~11GB)
- Resume support: Interrupted downloads will resume on next attempt

---

### 🎙️ SoulX-Singer Simple

Simple synthesizer with auto-preprocessing.

**Inputs:**
- `model`: SOULX_MODEL from loader
- `prompt_audio`: Reference singing voice (AUDIO noodle)
- `target_audio`: Target melody/score (AUDIO noodle)
- `prompt_language`: Mandarin/English/Cantonese
- `target_language`: Mandarin/English/Cantonese
- `control_mode`: `melody` (F0 contour) or `score` (MIDI notes)
- `vocal_sep_prompt`: Apply vocal separation to prompt
- `vocal_sep_target`: Apply vocal separation to target
- `auto_pitch_shift`: Auto-match pitch ranges
- `pitch_shift`: Manual pitch shift (-12 to +12 semitones)
- `n_steps`: Diffusion steps (16-64, default 32)
- `cfg_scale`: CFG guidance (1.0-5.0, default 3.0)

**Outputs:**
- `audio`: Generated singing voice (AUDIO)

**Note:** First run will download preprocessing models to `ComfyUI/models/SoulX-Singer/preprocessors/` if not already present.

---

### 🎙️ SoulX-Singer Advanced

Advanced synthesizer using pre-processed metadata files for manual editing workflows.

**Inputs:**
- `model`: SOULX_MODEL from loader
- `prompt_audio`: Reference audio (AUDIO noodle)
- `prompt_metadata_path`: Path to prompt JSON metadata file
- `target_metadata_path`: Path to target JSON metadata file
- `control_mode`: `melody` (F0 contour) or `score` (MIDI notes)
- `auto_pitch_shift`: Auto-match pitch ranges
- `pitch_shift`: Manual pitch shift (-12 to +12 semitones)
- `n_steps`: Diffusion steps (16-64, default 32)
- `cfg_scale`: CFG guidance (1.0-5.0, default 3.0)

**Outputs:**
- `audio`: Generated singing voice (AUDIO)

<details>
<summary><b>📋 Click to expand Metadata JSON Structure</b></summary>

```json
[
  {
    "index": "vocal_0_6900",
    "language": "English",
    "time": [0, 6900],
    "duration": "0.16 0.24 0.32...",
    "text": "<SP> Hello world <SP>...",
    "phoneme": "<SP> en_HH-ER0...",
    "note_pitch": "0 68 67 65...",
    "note_type": "1 2 2 2...",
    "f0": "0.0 0.0 382.7..."
  }
]
```

**Key Fields:**
- `time`: Segment boundaries [start_ms, end_ms]
- `text`: Lyrics with `<SP>` markers for word boundaries
- `phoneme`: ARPAbet phonemes (en_ prefix for English)
- `note_pitch`: MIDI note numbers (0=silence, 60=middle C)
- `note_type`: 1=rest, 2=sustain, 3=attack
- `f0`: Frame-level fundamental frequency in Hz

**Use Case:** 
1. Run Simple mode to get auto-generated metadata
2. Copy metadata files from temp folder (shown in console logs)
3. Edit in [MIDI Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor)
4. Use Advanced node with edited target metadata

**Why no target_audio input?**
The target is defined entirely by metadata - the node synthesizes new audio from scratch based on the metadata (lyrics, notes, timing). The prompt_audio provides the voice timbre reference.

</details>

---

## 📊 Parameters Explained

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| **model_name** | Model variant | `SoulX-Singer_model_bf16` (fast), `SoulX-Singer_model_fp32` (best quality) |
| **dtype** | Model precision | `bf16` (default, fast + quality), `fp32` (best quality) |
| **attention_type** | Attention mechanism | `sdpa` (default), `sageattention` (requires package) |
| **keep_loaded** | Cache model | `True` for multiple runs |
| **control_mode** | Pitch control | `melody` for natural, `score` for MIDI |
| **auto_pitch_shift** | Auto pitch matching | `True` for different singers |
| **n_steps** | Quality vs speed | `32` (balanced), `64` (best) |
| **cfg_scale** | Prompt adherence | `3.0` (balanced) |

---

## 🔧 Troubleshooting

<details>
<summary><b>🛠️ Click to expand troubleshooting guide</b></summary>

### Models Not Downloading?

Manually download from [drbaph/SoulX-Singer](https://huggingface.co/drbaph/SoulX-Singer):
```bash
pip install -U huggingface_hub
huggingface-cli download drbaph/SoulX-Singer --local-dir ComfyUI/models/SoulX-Singer
```

### Missing Dependencies?

Install all dependencies:
```bash
cd ComfyUI/custom_nodes/ComfyUI-SoulX-Singer
pip install -r requirements.txt
```

Common missing packages:
- `wandb` - for preprocessing logging
- `pretty_midi` - for MIDI handling
- `ml-collections` - for config management
- `loralib` - for LoRA model components
- `sageattention` - for optimized attention (optional, `pip install sageattention`)

### Out of Memory?

- Use `bf16` dtype instead of `fp32`
- Select `SoulX-Singer_model_bf16` instead of fp32
- Set `keep_loaded=False`
- Reduce `n_steps`
- Close other applications

### Slow Synthesis?

- Install SageAttention: `pip install sageattention`, then select `sageattention` attention type
- Use GPU with CUDA support
- Enable `keep_loaded=True`
- Use `bf16` dtype

### Preprocessing Pipeline Fails?

Check that all preprocessing models are downloaded to:
```
ComfyUI/models/SoulX-Singer/preprocessors/
```

Verify the directory structure matches the example above.

### SageAttention Not Working?

Make sure you have the sageattention package installed:
```bash
pip install sageattention
```

If you get errors with SageAttention, fall back to `sdpa` attention type.

</details>

---

## 🔗 Important Links

### 🤗 HuggingFace
- **Models & Preprocessors:** [drbaph/SoulX-Singer](https://huggingface.co/drbaph/SoulX-Singer/tree/main)
- **Online Demo:** [Soul-AILab/SoulX-Singer](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer)
- **Paper:** [huggingface.co/papers/2602.07803](https://huggingface.co/papers/2602.07803)

### 📄 Paper & Code
- **arXiv Paper:** [arxiv.org/abs/2602.07803](https://arxiv.org/abs/2602.07803)
- **Official Repository:** [Soul-AILab/SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer)
- **Demo Page:** [soul-ailab.github.io/soulx-singer](https://soul-ailab.github.io/soulx-singer/)

### 🛠️ Tools
- **MIDI Editor:** [Soul-AILab/SoulX-Singer-Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor)
- **ComfyUI Node:** [Saganaki22/ComfyUI-SoulX-Singer](https://github.com/Saganaki22/ComfyUI-SoulX-Singer)

---

## 📄 Citation

If you use SoulX-Singer in your research, please cite:

```bibtex
@misc{soulxsinger,
      title={SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis}, 
      author={Jiale Qian and Hao Meng and Tian Zheng and Pengcheng Zhu and Haopeng Lin and Yuhang Dai and Hanke Xie and Wenxiao Cao and Ruixuan Shang and Jun Wu and Hongmei Liu and Hanlin Wen and Jian Zhao and Zhonglin Jiang and Yong Chen and Shunshun Yin and Ming Tao and Jianguo Wei and Lei Xie and Xinsheng Wang},
      year={2026},
      eprint={2602.07803},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2602.07803}, 
}
```

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## ⚠️ Usage Disclaimer

SoulX-Singer is intended for academic research, educational purposes, and legitimate applications. Please use responsibly and ethically.

We advocate for the responsible development and use of AI and encourage the community to uphold safety and ethical principles in AI research and applications. If you have any concerns regarding ethics or misuse, please contact us.

---

<div align="center">
    <b><em>High-Quality Zero-Shot Singing Voice Synthesis for ComfyUI</em></b>
</div>
