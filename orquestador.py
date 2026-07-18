from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.globals import set_debug
from langgraph.prebuilt import create_react_agent

# --- IMPORTACIONES LOCALES ---
from my_models import GEMINI_FLASH, COHERE
from my_keys import GEMINI_API_KEY, COHERE_API_KEY
from herramienta_analisis_imagen import HerramientaAnalisisImagen
from herramienta_explicar import HerramientaExplicar

set_debug(False)


class AgenteOrquestador:
    def __init__(self):
        # 1. Instanciar el LLM
        self.llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_FLASH)

        # 2. Instanciar las herramientas
        self.herramientas = [HerramientaAnalisisImagen(), HerramientaExplicar()]

        # 3. Crear el agente usando LangGraph (El estándar moderno de LangChain)
        self.agent_executor = create_react_agent(self.llm, self.herramientas)

    def ejecutar(self, pregunta: str) -> str:
        """
        Ejecuta el agente pasándole la pregunta del usuario.
        """
        # LangGraph espera que le pasemos un historial de mensajes
        respuesta = self.agent_executor.invoke({"messages": [("user", pregunta)]})

        # El resultado incluye todos los pasos y mensajes.
        # La respuesta final de Gemini siempre será el último mensaje de la lista.
        mensaje_final = respuesta["messages"][-1].content

        return mensaje_final


# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    orquestador = AgenteOrquestador()

    # EJEMPLO 1: Usando ambas herramientas en una sola petición
    comando_compuesto = (
        "1. Analiza la imagen ubicada en datos/ejemplo_grafico.jpg. "
        "2. Luego, toma el concepto principal o las etiquetas que encontraste "
        "y utiliza tu herramienta de explicación para explicarme detalladamente qué significa ese concepto."
    )

    print(f"Enviando solicitud: '{comando_compuesto}'\n")
    resultado = orquestador.ejecutar(comando_compuesto)

    print("\n--- RESPUESTA DEL AGENTE ---")
    print(resultado)
