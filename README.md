# Voicer: Text-to-Speech Conversion Agent

Voicer is a powerful Python-based text-to-speech conversion tool that leverages the ElevenLabs API to transform text into high-quality audio files. It is designed to handle long texts gracefully with built-in chunking, supports model-specific character limits, and offers seamless integration with Jupyter notebooks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Direct Text Conversion](#direct-text-conversion)
  - [File-based Conversion](#file-based-conversion)
  - [Processing Long Texts](#processing-long-texts)
  - [Jupyter Notebook Integration](#jupyter-notebook-integration)
  - [Retrieving Voice Information](#retrieving-voice-information)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
- [Model Character Limits](#model-character-limits)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Error Handling](#error-handling)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Overview

Voicer is a robust text-to-speech conversion tool built in Python. It leverages the ElevenLabs API to generate high-quality audio from text while automatically handling long texts by splitting them into manageable chunks. With features like inline audio playback in Jupyter notebooks and detailed voice information retrieval, Voicer is designed for both developers and researchers looking for a versatile TTS solution.

## Features

- **High-Quality Audio Conversion:** Transform text directly into speech using state-of-the-art voices.
- **File Processing:** Convert entire text files to audio with automatic output format handling.
- **Automatic Chunking:** Seamlessly split texts that exceed model-specific character limits.
- **Jupyter Notebook Integration:** Enjoy inline audio playback and interactive voice information display.
- **Voice Information Retrieval:** Fetch detailed voice data such as name, labels, description, and preview URL.
- **Configurable Settings:** Easily customize voice, model, and output format parameters.
- **Automatic Output File Management:** Timestamp-based naming or custom file names.
- **Type-Safe Implementation:** Built with comprehensive error handling for robust performance.
- **Multiple Output Formats:** Supports various audio output configurations.

## Prerequisites

- **Python:** 3.8+ (up to Python 3.12 supported)
- **ElevenLabs API Key:** Required to authenticate API requests

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/evandeilton/voicer.git
    cd voicer
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Your ElevenLabs API Key:**

    You can set your API key as an environment variable:

    ```bash
    export ELEVENLABS_API_KEY='your_api_key_here'
    ```

    Alternatively, create a `.env` file in the project root:

    ```bash
    echo "ELEVENLABS_API_KEY=your_api_key_here" > .env
    ```

## Usage

### Direct Text Conversion

Convert a simple text string into an audio file:

```python
from voicer import TextToSpeechProcessor

processor = TextToSpeechProcessor()
output_path = processor.process_text("Hello, this is a test of text-to-speech conversion.")
print(f"Audio saved to: {output_path}")
```

### File-based Conversion

Convert the contents of a text file to audio:

```python
from voicer import TextToSpeechProcessor

processor = TextToSpeechProcessor()
output_path = processor.process_file("input/sample.txt")
print(f"Audio saved to: {output_path}")
```

### Processing Long Texts

For texts that exceed the model's character limit, Voicer automatically splits the text into chunks:

```python
from voicer import TextToSpeechProcessor

processor = TextToSpeechProcessor()
long_text = " ".join(["This is a long sentence."] * 500)
chunk_paths = processor.process_text_with_chunks(long_text)
print("Audio chunks saved to:", chunk_paths)
```

### Jupyter Notebook Integration

Display voice information and play audio inline within a Jupyter Notebook:

```python
from voicer import voicer

voicer(
    text="Hello from Jupyter!",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_flash_v2_5"
)
```

### Retrieving Voice Information

Retrieve and display detailed voice information:

```python
from voicer import VoiceInfoRetriever

retriever = VoiceInfoRetriever()
voice_info = retriever.get_voice_info("JBFqnCBsd6RMkjVDRZzb")
print(voice_info)
```

### Command-Line Interface (CLI)

Voicer also provides a CLI for converting text to speech. For example, to convert a text file:

```bash
python voicer.py --input input/sample.txt --voice JBFqnCBsd6RMkjVDRZzb --model eleven_flash_v2_5 --play
```

Alternatively, to convert a text string:

```bash
python voicer.py --text "Hello, world!" --output hello.mp3 --info
```

## Model Character Limits

Voicer automatically enforces model-specific character limits:

- **eleven_flash_v2_5:** 40,000 characters
- **eleven_flash_v2:** 30,000 characters
- **eleven_multilingual_v2:** 10,000 characters
- **eleven_multilingual_v1:** 10,000 characters
- **eleven_english_sts_v2:** 10,000 characters
- **eleven_english_sts_v1:** 10,000 characters

## Project Structure

- **`voicer.py`:** Core text-to-speech processing logic and CLI entry point.
- **`test_voicer.py`:** Test suite for the package.
- **`input/`:** Directory for input text files.
- **`output/`:** Directory for generated audio files.
- **`setup.py`:** Project packaging configuration.
- **`requirements.txt`:** List of package dependencies.

## Customization

Customize the voice, model, and output format by passing parameters to the `TextToSpeechProcessor`:

```python
processor = TextToSpeechProcessor(
    voice_id="custom_voice_id",
    model_id="eleven_flash_v2_5",
    output_format="mp3_44100_128"
)
```

## Error Handling

Voicer includes comprehensive error handling for:

- API key validation
- Character limit checks
- File I/O operations
- Audio playback issues
- Model compatibility and configuration errors

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

José Lopes  
Email: [evandeilton@gmail.com](mailto:evandeilton@gmail.com)  
Project Link: [https://github.com/evandeilton/voicer](https://github.com/evandeilton/voicer)

## Acknowledgments

- [ElevenLabs](https://elevenlabs.io/) for their excellent Text-to-Speech API.
- All contributors who have helped improve this package.
