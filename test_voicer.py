# """
# Test script for the Voicer package to demonstrate its functionality.
# """

# import os
# from pathlib import Path
# from typing import NoReturn

# from voicer import TextToSpeechProcessor, VoiceInfoRetriever, voicer, VoiceInfo


# def test_voicer() -> None:
#     """
#     Comprehensive test of the Voicer package functionality.
#     Requires ELEVENLABS_API_KEY to be set in the environment.
#     """
#     # Ensure API key is set
#     if not os.getenv("ELEVENLABS_API_KEY"):
#         print("Warning: ELEVENLABS_API_KEY is not set. Skipping live API tests.")
#         return

#     # Create input directory if it doesn't exist
#     input_dir = Path("input")
#     try:
#         input_dir.mkdir(exist_ok=True)
#     except Exception as e:
#         print(f"Error creating input directory: {e}")
#         return

#     # Test Voice Info Retrieval
#     print("\n--- Testing Voice Information Retrieval ---")
#     try:
#         retriever = VoiceInfoRetriever()
#         voice_id = "JBFqnCBsd6RMkjVDRZzb"  # Default voice ID
#         voice_info: VoiceInfo = retriever.get_voice_info(voice_id)
#         print("Voice Information Retrieved Successfully:")
#         for key, value in voice_info.items():
#             print(f"{key}: {value}")
#     except Exception as e:
#         print(f"Error retrieving voice info: {e}")

#     # Test Text-to-Speech Processing
#     print("\n--- Testing Text-to-Speech Processing ---")
#     try:
#         processor = TextToSpeechProcessor()
        
#         # Test short text processing
#         # short_text = "Hello! This is a test of the text-to-speech conversion."
#         # print("\nProcessing Short Text:")
#         # short_audio_path = processor.process_text(short_text, play_audio=False)
#         # print(f"Short text audio saved to: {short_audio_path}")
        
#         # # Test long text processing with chunks
#         # long_text = " ".join(["This is a long sentence to test chunk processing."] * 100)
#         # print("\nProcessing Long Text with Chunks:")
#         # chunk_paths = processor.process_text_with_chunks(long_text, play_audio=False)
#         # print("Chunk audio paths:")
#         # for path in chunk_paths:
#         #     print(f"  - {path}")
        
#         # Test file processing
#         print("\nTesting File Processing:")
#         sample_file = input_dir / "sample.txt"
#         # with open(sample_file, "w") as f:
#         #     f.write("Este é um exemplo de Texto para Voz do Eleven Labs.\n")
#         file_audio_path = processor.process_file(sample_file, play_audio=False)
#         print(f"File audio saved to: {file_audio_path}")

#         # Clean up test file
#         try:
#             sample_file.unlink()
#         except Exception as e:
#             print(f"Warning: Could not remove test file: {e}")
    
#     except Exception as e:
#         print(f"Error in text-to-speech processing: {e}")

#     # Optional: Demonstrate voicer function (commented out to avoid automatic audio play)
#     # print("\n--- Demonstrating Voicer Function ---")
#     # voicer(play_audio=False)


# def main() -> NoReturn:
#     """Main function to run the test script."""
#     test_voicer()


# if __name__ == "__main__":
#     main()