from setuptools import setup, find_packages

setup(
    name='voicer',
    version='0.2.0',  # Updated version for new improvements
    description='Text-to-Speech Conversion Agent using ElevenLabs API',
    long_description="""
    Voicer is a Python package that provides a robust interface to the ElevenLabs Text-to-Speech API.
    
    Features:
    - Easy-to-use text-to-speech conversion with multiple voices
    - Support for long text processing with automatic chunking
    - Voice information retrieval
    - Interactive Jupyter notebook interface
    - Automatic handling of model-specific character limits
    - File-based processing support
    """,
    long_description_content_type="text/markdown",
    author='José Lopes',
    author_email='evandeilton@gmail.com',
    packages=find_packages(),
    install_requires=[
        'python-dotenv>=0.21.0',
        'elevenlabs>=0.2.0',
        'typing-extensions>=4.0.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',  # Updated from Alpha to Beta
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'voicer=voicer:main',
        ],
    },
    project_urls={
        'Source': 'https://github.com/evandeilton/voicer',
        'Bug Reports': 'https://github.com/evandeilton/voicer/issues',
    },
    keywords=['text-to-speech', 'elevenlabs', 'tts', 'voice', 'audio', 'speech'],
)