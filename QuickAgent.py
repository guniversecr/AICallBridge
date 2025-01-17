import asyncio
from dotenv import load_dotenv
import shutil
import subprocess
import requests
import time
import os
from playsound import playsound

from pydub import AudioSegment
from pydub.playback import play

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
)

import requests
from pydub import AudioSegment
from io import BytesIO

import openai
from pathlib import Path
import pyttsx3


load_dotenv()

ELEVENLABS_API_KEY = "sk_5ea0181cb80cd4cc549279ca9b33384cb20a86be64326ce8"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Cambia por el ID de voz deseado de ElevenLabs
MODEL_NAME = "aura-helios-en"  # Puedes ajustar el nombre del modelo según tus necesidades.


class LanguageModelProcessor:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", groq_api_key=os.getenv("GROQ_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", openai_api_key=os.getenv("OPENAI_API_KEY"))
        # self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125", openai_api_key=os.getenv("OPENAI_API_KEY"))

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Load the system prompt from a file
        with open('system_prompt.txt', 'r') as file:
            system_prompt = file.read().strip()
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{text}")
        ])

        self.conversation = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )

    def process(self, text):
        self.memory.chat_memory.add_user_message(text)  # Add user message to memory

        start_time = time.time()

        # Go get the response from the LLM
        response = self.conversation.invoke({"text": text})
        end_time = time.time()

        self.memory.chat_memory.add_ai_message(response['text'])  # Add AI response to memory

        elapsed_time = int((end_time - start_time) * 1000)
        print(f"LLM ({elapsed_time}ms): {response['text']}")
        return response['text']

class TextToSpeech:
    # Set your Deepgram API Key and desired voice model
    DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    MODEL_NAME = "aura-helios-en"  # Example model name, change as needed

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text):
        if not self.is_installed("ffplay"):
            raise ValueError("ffplay not found, necessary to stream audio.")

        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        player_command = ["ffplay", "-autoexit", "-", "-nodisp"]
        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        start_time = time.time()  # Record the time before sending the request
        first_byte_time = None  # Initialize a variable to store the time when the first byte is received

        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:  # Check if this is the first chunk received
                        first_byte_time = time.time()  # Record the time when the first byte is received
                        ttfb = int((first_byte_time - start_time)*1000)  # Calculate the time to first byte
                        print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                    player_process.stdin.write(chunk)
                    player_process.stdin.flush()

        if player_process.stdin:
            player_process.stdin.close()
        player_process.wait()

    def text_to_speech_elevenlabs(self, text):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        data = {
            "text": text,
            # "model_id": "eleven_monolingual_v1",
            # "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        }

        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            # Guardar el audio en un archivo temporal
            audio_file = "output_audio.mp3"
            with open(audio_file, "wb") as f:
                f.write(response.content)
            
            # Usar pydub para reproducir el audio
            audio = AudioSegment.from_mp3(audio_file)
            play(audio)
            
            os.remove(audio_file)
        else:
            print("Error en la API de ElevenLabs:", response.status_code, response.text)

    def text_to_speech_deepgram(self, text):
        # Configura la URL de la API de Deepgram para la conversión de texto a voz
        
        DEEPGRAM_URL = f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&performance=some&encoding=linear16&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text
        }

        # Realiza la solicitud POST a la API de Deepgram
        response = requests.post(DEEPGRAM_URL, json=payload, headers=headers, stream=True)

        if response.status_code == 200:
            # Crear un archivo de audio temporal desde los datos de respuesta
            audio_data = BytesIO(response.content)
            
            # Convertir el audio en formato MP3
            audio = AudioSegment.from_file(audio_data, format="wav")  # Si es wav, puedes adaptarlo según el formato.
            
            # Reproducir el audio
            audio.export("output_audio.wav", format="wav")  # Puedes exportar a MP3 si lo prefieres
            play(audio)
        else:
            print("Error en la API de Deepgram:", response.status_code, response.text)

    def text_to_speech_openai(self, text): 
        openai.api_key = "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"  # Usa tu propia API Key
        speech_file_path = Path(__file__).parent / "speech.mp3"

        response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
        )
        response.stream_to_file(speech_file_path)


        print(response)
        # audio_data = BytesIO(response)
            
        # # Convertir el audio en formato MP3
        # audio = AudioSegment.from_file(audio_data, format="wav")  # Si es wav, puedes adaptarlo según el formato.
        
        # # Reproducir el audio
        # audio.export("output_audio.wav", format="wav")  # Puedes exportar a MP3 si lo prefieres
        # play(audio)

    def txt(self, text):
        print(text)
        
    def tts(self, text): 
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id) #change index to change voices
        engine.say(text)
        
        engine.runAndWait()

class TranscriptCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        self.transcript_parts = []

    def add_part(self, part):
        self.transcript_parts.append(part)

    def get_full_transcript(self):
        return ' '.join(self.transcript_parts)

transcript_collector = TranscriptCollector()

async def get_transcript(callback):
    transcription_complete = asyncio.Event()  # Event to signal transcription completion

    try:
        # example of setting up a client config. logging values: WARNING, VERBOSE, DEBUG, SPAM
        config = DeepgramClientOptions(options={"keepalive": "true","smart_format": "true",})
        deepgram: DeepgramClient = DeepgramClient("", config)

        dg_connection = deepgram.listen.asynclive.v("1")
        print ("Listening...")

        async def on_message(self, result, **kwargs):
            sentence = result.channel.alternatives[0].transcript
            
            if not result.speech_final:
                transcript_collector.add_part(sentence)
            else:
                # This is the final part of the current sentence
                transcript_collector.add_part(sentence)
                full_sentence = transcript_collector.get_full_transcript()
                # Check if the full_sentence is not empty before printing
                if len(full_sentence.strip()) > 0:
                    full_sentence = full_sentence.strip()
                    print(f"Human: {full_sentence}")
                    callback(full_sentence)  # Call the callback with the full_sentence
                    transcript_collector.reset()
                    transcription_complete.set()  # Signal to stop transcription and exit

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            punctuate=True,
            language="es",
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            endpointing=300,
            smart_format=True,
        )

        await dg_connection.start(options)

        # Open a microphone stream on the default input device
        microphone = Microphone(dg_connection.send)
        microphone.start()

        await transcription_complete.wait()  # Wait for the transcription to complete instead of looping indefinitely

        # Wait for the microphone to close
        microphone.finish()

        # Indicate that we've finished
        await dg_connection.finish()

    except Exception as e:
        print(f"Could not open socket: {e}")
        return

class ConversationManager:
    def __init__(self):
        self.transcription_response = ""
        self.llm = LanguageModelProcessor()

    async def main(self):
        def handle_full_sentence(full_sentence):
            self.transcription_response = full_sentence

        # Loop indefinitely until "goodbye" is detected
        while True:
            await get_transcript(handle_full_sentence)
            
            # Check for "goodbye" to exit the loop
            if "goodbye" in self.transcription_response.lower():
                break
            
            llm_response = self.llm.process(self.transcription_response)

            tts = TextToSpeech()
            #tts.speak(llm_response)
            
            #tts.txt(llm_response)

            tts.tts(llm_response)
            # tts.text_to_speech_openai(llm_response)
            # tts.text_to_speech_elevenlabs(llm_response)
            # tts.text_to_speech_deepgram(llm_response)
            # Reset transcription_response for the next loop iteration
            self.transcription_response = ""

if __name__ == "__main__":
    manager = ConversationManager()
    asyncio.run(manager.main())