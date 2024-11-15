from flask import Flask, render_template, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage
import requests
import tempfile
import os
from num2words import num2words
import locale
from dotenv import load_dotenv
import shutil
import speech_recognition as sr
from pydub import AudioSegment
import re
from tiktoken import get_encoding 

load_dotenv()

app = Flask(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
selected_voice_id = "PKygEn0yFu7wfoOxDsFB"

clientes = [
    {
        "ID_Cliente": 1,
        "Nombre_Cliente": "Luis Guillermo Pardo",
        "Fecha_Nacimiento": "1980-04-15",
        "Número_Documento": "123456789",
        "Teléfono_Contacto": "555-1234",
        "Correo_Electrónico": "luis.pardo@example.com",
        "Monto_Deuda": 1000000.00,
        "Fecha_Vencimiento": "2024-12-31",
        "Estado_Cuenta": "En mora",
        "Historial_Pagos": [
            {"Fecha": "2024-01-15", "Monto": 200000.00},
            {"Fecha": "2024-06-15", "Monto": 150000.00}
        ]
    },
    {
        "ID_Cliente": 2,
        "Nombre_Cliente": "María López",
        "Fecha_Nacimiento": "1990-07-22",
        "Número_Documento": "987654321",
        "Teléfono_Contacto": "555-5678",
        "Correo_Electrónico": "maria.lopez@example.com",
        "Monto_Deuda": 500000.00,
        "Fecha_Vencimiento": "2024-11-15",
        "Estado_Cuenta": "Pendiente",
        "Historial_Pagos": [
            {"Fecha": "2023-11-15", "Monto": 50000.00}
        ]
    },
    {
        "ID_Cliente": 3,
        "Nombre_Cliente": "Carlos Ramírez",
        "Fecha_Nacimiento": "1975-12-05",
        "Número_Documento": "111223344",
        "Teléfono_Contacto": "555-4321",
        "Correo_Electrónico": "carlos.ramirez@example.com",
        "Monto_Deuda": 1500000.00,
        "Fecha_Vencimiento": "2024-10-10",
        "Estado_Cuenta": "En mora",
        "Historial_Pagos": [
            {"Fecha": "2023-12-01", "Monto": 300000.00}
        ]
    }
]

public_audio_dir = 'static/audio'
if not os.path.exists(public_audio_dir):
    os.makedirs(public_audio_dir)

conversation_context = []  

TOKEN_COST = 0.002 / 1000  
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')

def calculate_costs(text):
    """Calcular palabras, tokens y costo estimado."""
    encoding = get_encoding("cl100k_base")
    tokens = len(encoding.encode(text))
    words = len(text.split())
    estimated_cost = tokens * TOKEN_COST
    return {"words": words, "tokens": tokens, "estimated_cost": round(estimated_cost, 6)}

def format_text_for_speech(text):
    def replace_numbers_with_words(match):
        number = int(match.group())
        return num2words(number, lang='es')  
    return re.sub(r'\b\d+\b', replace_numbers_with_words, text)

def get_gpt_response(prompt, cliente=None):
    """Obtener respuesta del modelo LLM con una guía estructurada de negociación."""
    context = ""
    if cliente:
        context = (
            "Ten muy presente que tienes que dar toda la información en español, sobre todo los numeros."
            f"Cliente: {cliente['Nombre_Cliente']}, con deuda de ${cliente['Monto_Deuda']:,.2f}, "
            f"fecha de vencimiento: {cliente['Fecha_Vencimiento']}, estado de cuenta: {cliente['Estado_Cuenta']}.\n"
            "Instrucciones para negociación:\n"
            "1. Si el cliente está dispuesto a negociar, ofrecer opciones de pago con tasas de interés reducidas o plan de pago mensual/trimestral.\n"
            "2. Si el cliente muestra resistencia a pagar, explicar los efectos de la deuda en su vida crediticia y consecuencias legales si la deuda sigue en mora.\n"
            "3. Si el cliente está indeciso, sugerir opciones de pago flexibles, destacando los beneficios de cumplir para mejorar su historial crediticio.\n"
            "Al finalizar la negociación, resume el acuerdo, indicando la tasa de interés, monto, plan de pago (mensual o trimestral) y otros detalles acordados.\n"
        )

    full_prompt = f"{context}\n{prompt}"
    try:
        message = HumanMessage(content=full_prompt)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error al generar respuesta del LLM: {e}")
        return "Lo siento, no puedo procesar tu solicitud en este momento."

def text_to_speech(text):
    """Convertir texto a audio usando ElevenLabs."""
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": eleven_api_key
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v1",
            "voice_settings": {
                "stability": 0.4,
                "similarity_boost": 1.0
            },
            "language": "es"
        }
        response = requests.post(url, json=data, headers=headers)

        if response.status_code != 200:
            print(f"Error al generar audio: {response.status_code}, {response.text}")
            return None

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    temp_file.write(chunk)
            temp_file.flush()

        public_audio_path = os.path.join(public_audio_dir, os.path.basename(temp_file.name))
        shutil.move(temp_file.name, public_audio_path)
        return f"/static/audio/{os.path.basename(temp_file.name)}"
    except Exception as e:
        print(f"Error en text-to-speech: {e}")
        return None

def audio_to_text(audio_file_path):
    """Convertir archivo de audio a texto."""
    recognizer = sr.Recognizer()
    wav_audio_path = audio_file_path.replace(".webm", ".wav")
    AudioSegment.from_file(audio_file_path).export(wav_audio_path, format="wav")

    with sr.AudioFile(wav_audio_path) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data, language="es-ES")



def analyze_sentiment(text, context):
    """Analizar el sentimiento usando el LLM en contexto."""
    full_context = "\n".join(context)
    prompt = (
        f"Analiza el sentimiento del texto más reciente en el contexto de la conversación:\n{full_context}\n"
        f"Texto más reciente: '{text}'.\n"
        "Proporciona el análisis en el siguiente formato organizado:\n"
        "Sentimiento: [positivo, negativo, neutral]\n"
        "Emoción dominante: [emoción principal, como preocupación, alegría, etc.]\n"
        "Indicador de negociación: [un número del 1 al 100, donde 1 significa fracaso total en la negociación y 100 significa un éxito completo]\n"
        "Acuerdos alcanzados:\n"
        "- [Enumerar en viñetas los acuerdos alcanzados o posibles recomendaciones para la negociación]\n"
        "Responde en español de manera clara y estructurada."
    )
    try:
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        print(f"Error al analizar el sentimiento: {e}")
        return "Análisis de sentimiento no disponible."

@app.route("/clientes", methods=["GET"])
def obtener_clientes():
    """Obtener lista básica de clientes."""
    return jsonify([
        {"ID_Cliente": cliente["ID_Cliente"], "Nombre_Cliente": cliente["Nombre_Cliente"], "Monto_Deuda": f"${cliente['Monto_Deuda']:,.2f}"}
        for cliente in clientes
    ])

@app.route("/cliente/<int:cliente_id>", methods=["GET"])
def obtener_cliente(cliente_id):
    """Obtener información detallada de un cliente."""
    cliente = next((c for c in clientes if c["ID_Cliente"] == cliente_id), None)
    if cliente:
        return jsonify(cliente)
    else:
        return jsonify({"error": "Cliente no encontrado"}), 404

@app.route("/interact", methods=["POST"])
def interact():
    """Manejar la interacción textual con el bot."""
    user_input = request.json.get('input', '').strip()
    cliente_id = request.json.get('cliente_id')

    if not user_input or not cliente_id:
        return jsonify({"error": "Faltan datos necesarios para procesar la solicitud."}), 400

    cliente = next((c for c in clientes if c["ID_Cliente"] == int(cliente_id)), None)
    response_text = get_gpt_response(user_input, cliente)
    conversation_context.append(f"Cliente: {user_input}")
    conversation_context.append(f"Val: {response_text}")


    formatted_text = format_text_for_speech(response_text)
    audio_path = text_to_speech(formatted_text)
    sentiment_analysis = analyze_sentiment(response_text, conversation_context)

    return jsonify({
        "text": response_text,
        "audio_path": audio_path,
        "sentiment_analysis": sentiment_analysis,
        "costs": calculate_costs(response_text),
    })

@app.route("/audio-interact", methods=["POST"])
def audio_interact():
    """Manejar interacción de audio con el bot."""
    audio = request.files['audio']
    cliente_id = request.form.get('cliente_id')  

    if not cliente_id:
        return jsonify({"error": "ID de cliente no proporcionado para la interacción de audio."}), 400

    cliente = next((c for c in clientes if c["ID_Cliente"] == int(cliente_id)), None)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio_file:
        audio.save(temp_audio_file.name)

    user_text = audio_to_text(temp_audio_file.name)
    conversation_context.append(f"Cliente: {user_text}")
    
    response_text = get_gpt_response(user_text, cliente)
    conversation_context.append(f"Val: {response_text}")

    formatted_text = format_text_for_speech(response_text)
    audio_path = text_to_speech(formatted_text)
    sentiment_analysis = analyze_sentiment(response_text, conversation_context)

    return jsonify({
        "user_text": user_text,
        "text": response_text,
        "audio_path": audio_path,
        "sentiment_analysis": sentiment_analysis,
        "costs": calculate_costs(response_text),
    })

@app.route("/reset", methods=["POST"])
def reset_conversation():
    """Reiniciar la conversación borrando el historial."""
    global conversation_context
    conversation_context = []  
    return jsonify({"message": "Conversación reiniciada exitosamente."})

@app.route("/conversation-history", methods=["GET"])
def download_conversation():
    """Descargar el historial de conversación en texto plano."""
    conversation_text = "\n".join(conversation_context)
    return jsonify({"conversation": conversation_text})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
