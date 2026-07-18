import ast
from langchain.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# --- IMPORTACIONES LOCALES ---
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from detalles_imagen import DetallesImagen

# --- INICIALIZACIÓN DEL MODELO ---
llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_FLASH)


# --- LÓGICA PRINCIPAL DE LA CADENA (LCEL) ---
def procesar_imagen_completa(ruta_imagen: str) -> dict:
    """
    Lee una imagen, la analiza con Gemini y luego formatea el resumen en JSON.
    """
    imagen = encode_image(ruta_imagen)

    template_analisis = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
        Asume que eres un analista de imágenes. Tu principal tarea consiste en: analizar una imagen
        para extraer las informaciones más relevantes de manera objetiva.

        #FORMATO DE SALIDA
        Descripción de la imagen: Tu descripción de la imagen aquí.
        Etiquetas: Una lista con 3 palabras clave separadas por comas.
        """,
            ),
            (
                "user",
                [
                    {"type": "text", "text": "Describe la imagen: "},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{imagen_informada}"
                        },
                    },
                ],
            ),
        ]
    )

    cadena_analisis = template_analisis | llm | StrOutputParser()
    parser_json = JsonOutputParser(pydantic_object=DetallesImagen)

    template_respuesta = PromptTemplate(
        template="""
        Genera un resumen, utilizando un lenguaje claro y objetivo, enfocado en el público colombiano. 
        La idea es que la comunicación del resultado sea lo más sencilla posible, priorizando los registros
        para consultas posteriores.

        #RESULTADO DE LA IMAGEN
        {respuesta_analisis_imagen}
        
        #FORMATO DE SALIDA
        {formato_salida}
        """,
        input_variables=["respuesta_analisis_imagen"],
        partial_variables={"formato_salida": parser_json.get_format_instructions()},
    )

    cadena_resumen = template_respuesta | llm | parser_json
    cadena_compuesta = {"respuesta_analisis_imagen": cadena_analisis} | cadena_resumen

    return cadena_compuesta.invoke({"imagen_informada": imagen})


# --- DEFINICIÓN DE HERRAMIENTAS ---
class HerramientaAnalisisImagen(BaseTool):
    name: str = "herramienta_analisis_imagen"
    description: str = """
    Utiliza esta herramienta para analizar una imagen y extraer las informaciones más relevantes de manera objetiva.
    
    #ENTRADAS REQUERIDAS
    - 'nombre_imagen' str: La ruta relativa de la imagen que se desea analizar, con extension .jpg o .png.
       ejemplo: 'datos/ejemplo_grafico.jpg'
    """
    return_direct: bool = False

    def _run(self, accion: str) -> str:
        try:
            accion_dict = ast.literal_eval(accion)
            nombre_imagen = accion_dict.get("nombre_imagen", accion)
        except (ValueError, SyntaxError):
            nombre_imagen = accion

        # CORRECCIÓN: Llamamos a la función real que hace el trabajo pesado
        try:
            resultado = procesar_imagen_completa(nombre_imagen)
            return str(resultado)
        except Exception as e:
            return f"Hubo un error al procesar la imagen en la ruta {nombre_imagen}: {str(e)}"


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    ruta = "datos/ejemplo_grafico.jpg"
    print(f"Iniciando análisis directo para: {ruta}...")
    try:
        resultado_final = procesar_imagen_completa(ruta)
        print("\n--- RESULTADO FINAL EN JSON ---")
        print(resultado_final)
    except Exception as e:
        print(f"Error en prueba directa: {e}")
