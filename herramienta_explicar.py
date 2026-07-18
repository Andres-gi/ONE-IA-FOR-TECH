import ast
from langchain.tools import BaseTool
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- IMPORTACIONES LOCALES ---
from my_keys import COHERE_API_KEY

# --- INICIALIZACIÓN DEL MODELO ---
llm_cohere = ChatCohere(cohere_api_key=COHERE_API_KEY)


class HerramientaExplicar(BaseTool):
    name: str = "herramienta_explicar"
    description: str = """
    Utiliza esta herramienta para explicar detalladamente el significado de uno o varios conceptos, palabras o etiquetas.
    
    #ENTRADAS REQUERIDAS
    - 'concepto' str: La palabra, etiqueta o frase que se desea explicar.
    """
    return_direct: bool = False

    def _run(self, accion: str) -> str:
        # 1. PARSEO ROBUSTO: Manejar tanto diccionarios en string como texto plano
        try:
            accion_dict = ast.literal_eval(accion)
            # Si es un dict, extraemos la llave (asumiendo que el LLM usó la llave 'concepto' o similar)
            if isinstance(accion_dict, dict):
                concepto_a_explicar = accion_dict.get("concepto", accion)
            else:
                concepto_a_explicar = str(accion_dict)
        except (ValueError, SyntaxError):
            # Si falla, significa que el LLM envió la frase directamente
            concepto_a_explicar = accion

        # 2. LÓGICA DE EXPLICACIÓN CON COHERE
        template_explicacion = PromptTemplate(
            template="""
            Eres un experto educador. Explica de manera detallada, clara y fácil de entender 
            el siguiente concepto o conjunto de etiquetas:
            
            Concepto a explicar: {concepto}
            """,
            input_variables=["concepto"],
        )

        cadena_explicar = template_explicacion | llm_cohere | StrOutputParser()

        try:
            # Invocamos la cadena de Cohere
            resultado = cadena_explicar.invoke({"concepto": concepto_a_explicar})
            return resultado
        except Exception as e:
            return f"Error al intentar explicar el concepto: {str(e)}"
