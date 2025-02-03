"""
Module: voicer
This module integrates functionalities for text-to-speech conversion and voice information retrieval
using the ElevenLabs API.

Features:
- Retrieve voice information (e.g., name, labels, preview URL) via the 'Get Voice' endpoint.
- Convert text to speech with support for model-specific character limits and chunk processing.
- Provide a direct API demonstration.
- Offer an interactive interface (voicer()) for use in notebooks (e.g., Jupyter).
"""

import os
import logging
import argparse       # Added for CLI parsing
import sys            # Added for sys.exit
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, TypedDict, Dict

import dotenv
from elevenlabs import stream, play
from elevenlabs.client import ElevenLabs

# Attempt to import IPython display for notebook inline audio support.
try:
    from IPython.display import Audio, display
except ImportError:
    Audio = None

# Load environment variables (e.g., ELEVENLABS_API_KEY)
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class VoiceInfo(TypedDict):
    """Type definition for voice information returned by the API."""
    name: str
    labels: Dict[str, str]
    description: Optional[str]
    preview_url: Optional[str]
    category: str


# Define character limits for supported models.
MODEL_CHARACTER_LIMITS: Dict[str, int] = {
    "eleven_flash_v2_5": 40000,
    "eleven_flash_v2": 30000,
    "eleven_multilingual_v2": 10000,
    "eleven_multilingual_v1": 10000,
    "eleven_english_sts_v2": 10000,
    "eleven_english_sts_v1": 10000,
}


def get_elevenlabs_client(api_key: Optional[str] = None) -> ElevenLabs:
    """
    Retrieve an instance of the ElevenLabs API client.
    
    Args:
        api_key (Optional[str]): API key for authentication. If not provided, it is loaded from the environment.
        
    Returns:
        ElevenLabs: An instance of the ElevenLabs client.
    
    Raises:
        ValueError: If the API key is not provided or found in the environment.
    """
    if not api_key:
        api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise ValueError("ELEVENLABS_API_KEY is required. Please set it in your environment or pass it explicitly.")
    logger.debug("ElevenLabs client initialized with provided API key.")
    return ElevenLabs(api_key=api_key)


class VoiceInfoRetriever:
    """
    A class to handle retrieval of voice information using the ElevenLabs API.
    """
    def __init__(self, client: Optional[ElevenLabs] = None) -> None:
        """
        Initialize the VoiceInfoRetriever.
        
        Args:
            client (Optional[ElevenLabs]): An instance of the ElevenLabs API client. If None, one is created.
        """
        self.client = client if client is not None else get_elevenlabs_client()
        
    def get_voice_info(self, voice_id: str) -> Dict:
        """
        Retrieve detailed information for a specific voice.
        
        Args:
            voice_id (str): The unique identifier of the voice to retrieve.
            
        Returns:
            Dict: A dictionary containing voice information.
            
        Raises:
            RuntimeError: If the API call fails.
        """
        logger.info(f"Retrieving voice information for voice_id '{voice_id}'...")
        try:
            voice_info = self.client.voices.get(voice_id=voice_id)
            info_dict = vars(voice_info)
            logger.info("Voice information retrieved successfully.")
            return info_dict
        except Exception as error:
            logger.exception(f"Failed to retrieve voice information for voice_id '{voice_id}'.")
            raise RuntimeError(f"Failed to retrieve voice information for voice_id '{voice_id}': {error}")


class TextToSpeechProcessor:
    """
    A class to handle text-to-speech conversion using the ElevenLabs API.
    
    It includes functionality to:
      - Check and enforce model-specific character limits.
      - Process text files.
      - Split long texts into manageable chunks.
    """
    
    def __init__(
        self,
        client: Optional[ElevenLabs] = None,
        voice_id: str = "YU8EsJtXFMyKMxYtheDk",  # Estive New
        model_id: str = "eleven_flash_v2_5",     # or "eleven_multilingual_v2"
        output_format: str = "mp3_44100_128"
    ):
        """
        Initialize the TextToSpeechProcessor.
        
        Args:
            client (Optional[ElevenLabs]): An instance of the ElevenLabs API client. If None, one is created.
            voice_id (str): The ID of the voice to use for conversion.
            model_id (str): The ID of the model to use for conversion.
            output_format (str): The format of the output audio file.
            
        Raises:
            ValueError: If the API key is missing or if the model_id is not supported.
        """
        self.client = client if client is not None else get_elevenlabs_client()
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        
        # Determine the maximum allowed characters for the chosen model.
        self.max_chars = MODEL_CHARACTER_LIMITS.get(model_id)
        if not self.max_chars:
            raise ValueError(f"Unsupported model_id: {model_id}")
        
        # Ensure the output directory exists.
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"TextToSpeechProcessor initialized with voice_id '{voice_id}', model_id '{model_id}', "
                    f"output_format '{output_format}', max_chars {self.max_chars}")
        
    def process_text(
        self,
        text: str,
        output_filename: Optional[str] = None,
        play_audio: bool = False
    ) -> Path:
        """
        Convert text to speech and save the audio file.
        
        Args:
            text (str): The text to convert to speech.
            output_filename (Optional[str]): The name of the output file. If not provided, a timestamp-based name is used.
            play_audio (bool): Whether to play the audio after conversion.
            
        Returns:
            Path: The path to the saved audio file.
            
        Raises:
            ValueError: If the text is empty or exceeds the model's character limit.
            RuntimeError: If the API conversion fails.
        """
        logger.debug("Starting text-to-speech conversion process...")
        if not text.strip():
            logger.error("Empty text provided for conversion.")
            raise ValueError("Text cannot be empty")
        
        # Check if text exceeds the allowed character limit for the model.
        if len(text) > self.max_chars:
            error_msg = (f"Text exceeds the character limit of {self.max_chars} for model '{self.model_id}'. "
                         "Consider using 'process_text_with_chunks' to process long texts.")
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info("Converting text to speech...")
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format=self.output_format
            )
            
            # Convert generator to bytes
            audio = b''.join(chunk for chunk in audio_generator)
            logger.debug("Audio conversion completed, received audio bytes.")
            
            # Generate a filename if none is provided.
            if output_filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tts_{timestamp}.mp3"
            else:
                # Ensure the filename has the correct extension.
                if not output_filename.endswith('.mp3'):
                    output_filename += '.mp3'
            
            output_path = self.output_dir / output_filename
            logger.info(f"Saving audio to file: {output_path}")
            
            # Save the audio file.
            with open(output_path, 'wb') as audio_file:
                audio_file.write(audio)
            logger.info("Audio file saved successfully.")
            
            if play_audio:
                try:
                    logger.info("Attempting to play the audio...")
                    play(audio)
                except Exception as e:
                    logger.warning(f"Failed to play audio: {e}")
            
            return output_path
        
        except Exception as e:
            logger.exception("Failed to convert text to speech.")
            raise RuntimeError(f"Failed to convert text to speech: {str(e)}")
    
    def process_file(
        self,
        input_file: Union[str, Path],
        play_audio: bool = False
    ) -> Path:
        """
        Convert text from a file to speech and save the audio file.
        
        Args:
            input_file (Union[str, Path]): Path to the input text file.
            play_audio (bool): Whether to play the audio after conversion.
            
        Returns:
            Path: The path to the saved audio file.
            
        Raises:
            FileNotFoundError: If the input file doesn't exist.
            RuntimeError: If file reading or conversion fails.
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        try:
            logger.info(f"Reading text from file: {input_path}")
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            output_filename = f"{input_path.stem}.mp3"
            return self.process_text(text, output_filename, play_audio)
            
        except Exception as e:
            logger.exception(f"Failed to process file: {input_file}")
            raise RuntimeError(f"Failed to process file {input_file}: {str(e)}")
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a long text into smaller chunks that do not exceed the model's character limit.
        
        Args:
            text (str): The text to be split.
            
        Returns:
            List[str]: A list of text chunks.
        """
        logger.debug("Splitting text into manageable chunks...")
        if len(text) <= self.max_chars:
            logger.debug("No need to split text; text length within limit.")
            return [text]
        
        words = text.split()
        chunks = []
        current_chunk = ""
        
        for word in words:
            # +1 accounts for a space between words.
            if len(current_chunk) + len(word) + (1 if current_chunk else 0) <= self.max_chars:
                current_chunk = f"{current_chunk} {word}".strip()
            else:
                chunks.append(current_chunk)
                current_chunk = word
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.info(f"Text split into {len(chunks)} chunks.")
        return chunks
    
    def process_text_with_chunks(
        self,
        text: str,
        base_output_filename: Optional[str] = None,
        play_audio: bool = False
    ) -> List[Path]:
        """
        Process a long text by splitting it into chunks and converting each chunk to speech separately.
        
        Args:
            text (str): The text to convert.
            base_output_filename (Optional[str]): Base name for the output files. Each chunk is saved as <base>_chunk<number>.mp3.
            play_audio (bool): Whether to play the audio for each chunk.
            
        Returns:
            List[Path]: A list of paths to the saved audio files.
        """
        logger.info("Processing text in chunks due to length constraints...")
        chunks = self.split_text(text)
        output_paths = []
        
        if base_output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_output_filename = f"tts_{timestamp}"
        else:
            base_output_filename = base_output_filename.rsplit('.', 1)[0]
        
        for idx, chunk in enumerate(chunks, start=1):
            filename = f"{base_output_filename}_chunk{idx}.mp3"
            logger.debug(f"Processing chunk {idx} with filename {filename}...")
            output_path = self.process_text(chunk, output_filename=filename, play_audio=play_audio)
            output_paths.append(output_path)
        
        logger.info("All text chunks processed successfully.")
        return output_paths


def voicer(
    text: str = "The first move is what sets everything in motion.",
    voice_id: str = "YU8EsJtXFMyKMxYtheDk",
    model_id: str = "eleven_flash_v2_5",
    output_format: str = "mp3_44100_128",
    play_audio: bool = True,
    api_key: Optional[str] = None,
    output_filename: Optional[str] = None
) -> None:
    """
    An interactive interface for Jupyter notebooks.
    
    This function retrieves voice information and processes text-to-speech conversion,
    displaying the voice details and playing the audio inline (if in a notebook environment).
    
    Args:
        text (str): The text to convert to speech.
        voice_id (str): The ID of the voice to use.
        model_id (str): The model to be used for conversion.
        output_format (str): The audio output format.
        play_audio (bool): Whether to automatically play the audio.
        api_key (Optional[str]): Your ElevenLabs API key. If None, the key is loaded from the environment.
        output_filename (Optional[str]): The desired output file name for the generated audio.
    """
    logger.info("Starting voicer interactive session...")
    client = get_elevenlabs_client(api_key)
    
    # Retrieve and display voice information.
    retriever = VoiceInfoRetriever(client=client)
    voice_info = retriever.get_voice_info(voice_id=voice_id)
    logger.info("Voice Information:")
    for key, value in voice_info.items():
        logger.info(f"{key}: {value}")
    
    # Convert text to speech.
    tts_processor = TextToSpeechProcessor(client=client, voice_id=voice_id, model_id=model_id, output_format=output_format)
    audio_path = tts_processor.process_text(text, output_filename=output_filename, play_audio=False)
    logger.info(f"Audio saved to: {audio_path}")
    
    # If in a notebook, display the audio inline.
    if Audio is not None:
        try:
            logger.info("Displaying audio inline in the notebook...")
            display(Audio(filename=str(audio_path)))
        except Exception as e:
            logger.warning(f"Could not display audio inline: {e}")
    else:
        logger.info("IPython.display.Audio not available; please open the audio file manually.")
    
    # Optionally play the audio using the ElevenLabs play function.
    if play_audio:
        try:
            logger.info("Playing audio via ElevenLabs play function...")
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            play(audio_data)
        except Exception as e:
            logger.warning(f"Warning: Failed to play audio: {e}")


def main() -> None:
    """
    CLI entry point for the ElevenLabs text-to-speech converter.
    
    This function parses command-line arguments to:
      - Accept a text string (--text) or a file path (--input) as the source text.
      - Set the voice ID, model ID, output format, and output file name.
      - Optionally display voice information and play the audio after conversion.
    """
    parser = argparse.ArgumentParser(
        description="CLI for ElevenLabs text-to-speech conversion."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to convert to speech. Use this OR --input."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to a text file to convert to speech. Use this OR --text."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output audio file name. If omitted, a timestamp-based name is generated."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="YU8EsJtXFMyKMxYtheDk",
        help="Voice ID to use for conversion."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="eleven_flash_v2_5",
        help="Model ID to use for conversion."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="mp3_44100_128",
        help="Output audio format."
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play the audio after conversion."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Your ElevenLabs API key. If omitted, the key is loaded from the environment."
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display voice information before conversion."
    )

    args = parser.parse_args()

    # Ensure that either --text or --input is provided.
    if not args.text and not args.input:
        parser.error("Either --text or --input must be provided.")

    # Read input text from file if provided; otherwise, use the --text argument.
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input}")
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.error(f"Failed to read input file: {e}")
            sys.exit(1)
    else:
        text = args.text

    # Initialize the ElevenLabs client.
    try:
        client = get_elevenlabs_client(api_key=args.api_key)
    except Exception as e:
        logging.error(f"Failed to initialize ElevenLabs client: {e}")
        sys.exit(1)

    # Display voice information if requested.
    if args.info:
        try:
            voice_retriever = VoiceInfoRetriever(client=client)
            voice_info = voice_retriever.get_voice_info(voice_id=args.voice)
        except Exception as e:
            logging.error(f"Failed to retrieve voice information: {e}")
            sys.exit(1)
        logging.info("Voice Information:")
        for key, value in voice_info.items():
            logging.info(f"  {key}: {value}")

    # Create the text-to-speech processor.
    try:
        tts_processor = TextToSpeechProcessor(
            client=client,
            voice_id=args.voice,
            model_id=args.model,
            output_format=args.format
        )
    except Exception as e:
        logging.error(f"Failed to initialize TextToSpeechProcessor: {e}")
        sys.exit(1)

    # Process the text. If it exceeds the model's character limit, process it in chunks.
    try:
        if len(text) > tts_processor.max_chars:
            logging.info("Input text exceeds maximum character limit. Processing in chunks...")
            output_paths = tts_processor.process_text_with_chunks(
                text,
                base_output_filename=args.output,
                play_audio=args.play
            )
            for path in output_paths:
                logging.info(f"Chunk audio saved to: {path}")
        else:
            audio_path = tts_processor.process_text(
                text,
                output_filename=args.output,
                play_audio=args.play
            )
            logging.info(f"Audio saved to: {audio_path}")
    except Exception as e:
        logging.exception(f"Failed to convert text to speech: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
