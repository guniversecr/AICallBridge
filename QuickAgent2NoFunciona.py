# import openai
# import requests
# import os
# import time
# import speech_recognition as sr
# from playsound import playsound

# # Configuración de API
# openai.api_key = "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"  # Usa tu propia API Key
# ELEVENLABS_API_KEY = "sk_5ea0181cb80cd4cc549279ca9b33384cb20a86be64326ce8"
# VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Cambia por el ID de voz deseado de ElevenLabs

# # Función para obtener respuestas desde un LLM
# conversation_history = [
#     {"role": "system", "content": "Eres un asistente útil y servicial."}
# ]

# def get_llm_response(prompt):
#     try:
#         # Agregar el mensaje del usuario al historial
#         conversation_history.append({"role": "user", "content": prompt})

#         # Utiliza la versión antigua de OpenAI API para obtener la respuesta
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",  # Asegúrate de que el modelo está correctamente definido
#             messages=conversation_history,
#             temperature=0.7,
#         )

#         # Extraer la respuesta del asistente
#         assistant_response = response['choices'][0]['message']['content']

#         # Agregar la respuesta del asistente al historial
#         conversation_history.append({"role": "assistant", "content": assistant_response})

#         return assistant_response
#     except Exception as e:
#         return f"Error al generar respuesta: {e}"

# # Función para convertir texto a voz usando ElevenLabs
# def text_to_speech_elevenlabs(text):
#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
#     headers = {
#         "Accept": "audio/mpeg",
#         "Content-Type": "application/json",
#         "xi-api-key": ELEVENLABS_API_KEY,
#     }
#     data = {
#         "text": text,
#         "model_id": "eleven_monolingual_v1",
#         "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
#     }

#     response = requests.post(url, json=data, headers=headers)
#     if response.status_code == 200:
#         # Guardar el audio en un archivo temporal
#         audio_file = "output_audio.mp3"
#         with open(audio_file, "wb") as f:
#             f.write(response.content)
#         # Reproducir el audio
#         playsound(audio_file)
#         os.remove(audio_file)
#     else:
#         print("Error en la API de ElevenLabs:", response.status_code, response.text)

# # Función para reconocimiento de voz
# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Escuchando...")
#         try:
#             audio = recognizer.listen(source)
#             text = recognizer.recognize_google(audio, language="es-ES")  # Cambia "es-ES" por otro idioma si es necesario
#             print(f"Tú: {text}")
#             return text
#         except sr.UnknownValueError:
#             print("No se entendió el audio.")
#             return None
#         except sr.RequestError as e:
#             print(f"Error con el servicio de reconocimiento de voz: {e}")
#             return None

# # Ciclo principal del agente conversacional
# def conversational_agent():
#     print("Iniciando el agente conversacional. Di 'adiós' para salir.")
#     while True:
#         # Capturar la entrada del usuario por voz
#         user_input = recognize_speech()
#         if not user_input:
#             continue
        
#         # Salir si el usuario dice "adiós"
#         if "adiós" in user_input.lower():
#             text_to_speech_elevenlabs("¡Adiós! Espero hablar contigo pronto.")
#             break
        
#         # Obtener respuesta del LLM
#         response = get_llm_response(user_input)
#         print(f"Agente: {response}")
        
#         # Convertir la respuesta en audio usando ElevenLabs
#         text_to_speech_elevenlabs(response)

# # Ejecutar el agente conversacional
# if __name__ == "__main__":
#     conversational_agent()
