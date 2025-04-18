{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/DavidLarreta/AI-Agent-with-llama3.1/blob/main/KI_Agent.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pLsGxbynxW_y"
      },
      "outputs": [],
      "source": [
        "!sudo apt-get install -y pciutils\n",
        "!curl -fsSL https://ollama.com/install.sh | sh # download ollama api\n",
        "\n",
        "from IPython.display import clear_output\n",
        "\n",
        "# Create a Python script to start the Ollama API server in a separate thread\n",
        "\n",
        "import os\n",
        "import threading\n",
        "import subprocess\n",
        "import requests\n",
        "import json\n",
        "\n",
        "def ollama():\n",
        "    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'\n",
        "    os.environ['OLLAMA_ORIGINS'] = '*'\n",
        "    subprocess.Popen([\"ollama\", \"serve\"])\n",
        "\n",
        "ollama_thread = threading.Thread(target=ollama)\n",
        "ollama_thread.start()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "--V_nMaNyfuY"
      },
      "outputs": [],
      "source": [
        "from IPython.display import clear_output\n",
        "!ollama pull llama3.1:8b\n",
        "clear_output()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AedVVvXIy-w6"
      },
      "outputs": [],
      "source": [
        "!pip install -U lightrag[ollama]\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KgXhaLrtzlzO"
      },
      "outputs": [],
      "source": [
        "from lightrag.core.generator import Generator\n",
        "from lightrag.core.component import Component\n",
        "from lightrag.core.model_client import ModelClient\n",
        "from lightrag.components.model_client import OllamaClient, GroqAPIClient\n",
        "\n",
        "import time\n",
        "\n",
        "\n",
        "qa_template = r\"\"\"<SYS>\n",
        "Eres mi asistente para mi presentacion de la noche IB, en donde voy a presentar a mi proyecto de evalucacion interna de informatica, voy a hablar primero sobre\n",
        "de que se trata el proyecto. El proyecto es una aplicacion/sitio web que se usa para la empresa Larreta Glass, es una aplicacion que se especializa en la reservacion de citas, cuenta con 4 pestañas, el inicio,\n",
        "testimonios, reservación y tambien cuenta con una opcion de cambiar a administrador, ademas en la pestaña de reservacion se observa un calendario que se usa para que los usuarios puedan reservar bien sus citas.\n",
        "El proyecto cuenta con los Criterios A, B, C, D y E, el A es de planificacion, el B es de diseño y organizacion, C es el que muestra que hace cada parte del codigo, el D es un video sobre el programa y el E es sobre las cosas para mejorar el programa.\n",
        "La nota que obtuve fue de 30/34 lo que se interpreta como un sobresaliente en el IB, puesto una nota alta es de 27 para arriba.\n",
        "Las dificultades que tuve en el proyecto fueron las siguientes: El pip no me funciono, la falta de organizacion fue notoria, falta de informacion a la hora de iniciar.\n",
        "Al inicio tienes que presentarte como mi asistente y mencionar si tienen alguna pregunta sobre mi presentación y que cualquier cosa puede ayudar, si no puedes responder una pregunta solo di que no puedes y que me pregunta a mi.\n",
        "\n",
        "</SYS>\n",
        "User: {{input_str}}\n",
        "You:\"\"\"\n",
        "\n",
        "class SimpleQA(Component):\n",
        "    def __init__(self, model_client: ModelClient, model_kwargs: dict):\n",
        "        super().__init__()\n",
        "        self.generator = Generator(\n",
        "            model_client=model_client,\n",
        "            model_kwargs=model_kwargs,\n",
        "            template=qa_template,\n",
        "        )\n",
        "\n",
        "    def call(self, input: dict) -> str:\n",
        "        return self.generator.call({\"input_str\": str(input)})\n",
        "\n",
        "    async def acall(self, input: dict) -> str:\n",
        "        return await self.generator.acall({\"input_str\": str(input)})"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iHihRZs7uMH2"
      },
      "outputs": [],
      "source": [
        "# Definir el historial de la conversación\n",
        "conversation_history = \"\"\n",
        "\n",
        "# Función para interactuar con el modelo y mantener la conversación\n",
        "def interact_with_model(question, conversation_history, qa):\n",
        "    # Añadir la nueva pregunta al historial\n",
        "    conversation_history += f\"User: {question}\\n\"\n",
        "\n",
        "    # Pasar el historial completo al modelo\n",
        "    output = qa(conversation_history)\n",
        "\n",
        "    # Añadir la respuesta del modelo al historial\n",
        "    conversation_history += f\"Model: {output.data}\\n\"\n",
        "\n",
        "    # Mostrar la respuesta del modelo\n",
        "    display(Markdown(f\"**Answer:** {output.data}\"))\n",
        "\n",
        "    return conversation_history\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8i3in9yC08Vn"
      },
      "outputs": [],
      "source": [
        "from lightrag.components.model_client import OllamaClient\n",
        "from IPython.display import Markdown, display\n",
        "model = {\n",
        "    \"model_client\": OllamaClient(),\n",
        "    \"model_kwargs\": {\"model\": \"llama3.1:8b\"}\n",
        "}\n",
        "qa = SimpleQA(**model)\n",
        "output=qa(\"Hola\")\n",
        "display(Markdown(f\"**Answer:** {output.data}\"))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "TaoXcW3NvI_o"
      },
      "outputs": [],
      "source": [
        "# Inicializar el modelo\n",
        "model = {\n",
        "    \"model_client\": OllamaClient(),\n",
        "    \"model_kwargs\": {\"model\": \"llama3.1:8b\"}\n",
        "}\n",
        "qa = SimpleQA(**model)\n",
        "\n",
        "while True:\n",
        "  entrada_usuario = input(\"Usuario:\")\n",
        "  conversation_history = interact_with_model(entrada_usuario, conversation_history, qa)\n",
        "\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPMovY4CQYDhelurPn/vX0p",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}