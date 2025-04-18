# -*- coding: utf-8 -*-
"""KI-Agent.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PxbDMnHsncvwuwl6HrnIPZPIYCkrTsKT
"""

!sudo apt-get install -y pciutils
!curl -fsSL https://ollama.com/install.sh | sh # download ollama api

from IPython.display import clear_output

# Create a Python script to start the Ollama API server in a separate thread

import os
import threading
import subprocess
import requests
import json

def ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()

from IPython.display import clear_output
!ollama pull llama3.1:8b
clear_output()

!pip install -U lightrag[ollama]

from lightrag.core.generator import Generator
from lightrag.core.component import Component
from lightrag.core.model_client import ModelClient
from lightrag.components.model_client import OllamaClient, GroqAPIClient

import time


qa_template = r"""<SYS>
Eres mi asistente para mi presentacion de la noche IB, en donde voy a presentar a mi proyecto de evalucacion interna de informatica, voy a hablar primero sobre
de que se trata el proyecto. El proyecto es una aplicacion/sitio web que se usa para la empresa Larreta Glass, es una aplicacion que se especializa en la reservacion de citas, cuenta con 4 pestañas, el inicio,
testimonios, reservación y tambien cuenta con una opcion de cambiar a administrador, ademas en la pestaña de reservacion se observa un calendario que se usa para que los usuarios puedan reservar bien sus citas.
El proyecto cuenta con los Criterios A, B, C, D y E, el A es de planificacion, el B es de diseño y organizacion, C es el que muestra que hace cada parte del codigo, el D es un video sobre el programa y el E es sobre las cosas para mejorar el programa.
La nota que obtuve fue de 30/34 lo que se interpreta como un sobresaliente en el IB, puesto una nota alta es de 27 para arriba.
Las dificultades que tuve en el proyecto fueron las siguientes: El pip no me funciono, la falta de organizacion fue notoria, falta de informacion a la hora de iniciar.
Al inicio tienes que presentarte como mi asistente y mencionar si tienen alguna pregunta sobre mi presentación y que cualquier cosa puede ayudar, si no puedes responder una pregunta solo di que no puedes y que me pregunta a mi.

</SYS>
User: {{input_str}}
You:"""

class SimpleQA(Component):
    def __init__(self, model_client: ModelClient, model_kwargs: dict):
        super().__init__()
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=qa_template,
        )

    def call(self, input: dict) -> str:
        return self.generator.call({"input_str": str(input)})

    async def acall(self, input: dict) -> str:
        return await self.generator.acall({"input_str": str(input)})

# Definir el historial de la conversación
conversation_history = ""

# Función para interactuar con el modelo y mantener la conversación
def interact_with_model(question, conversation_history, qa):
    # Añadir la nueva pregunta al historial
    conversation_history += f"User: {question}\n"

    # Pasar el historial completo al modelo
    output = qa(conversation_history)

    # Añadir la respuesta del modelo al historial
    conversation_history += f"Model: {output.data}\n"

    # Mostrar la respuesta del modelo
    display(Markdown(f"**Answer:** {output.data}"))

    return conversation_history

from lightrag.components.model_client import OllamaClient
from IPython.display import Markdown, display
model = {
    "model_client": OllamaClient(),
    "model_kwargs": {"model": "llama3.1:8b"}
}
qa = SimpleQA(**model)
output=qa("Hola")
display(Markdown(f"**Answer:** {output.data}"))

# Inicializar el modelo
model = {
    "model_client": OllamaClient(),
    "model_kwargs": {"model": "llama3.1:8b"}
}
qa = SimpleQA(**model)

while True:
  entrada_usuario = input("Usuario:")
  conversation_history = interact_with_model(entrada_usuario, conversation_history, qa)