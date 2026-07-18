from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from detalles_imagen import DetallesImagen
from my_models import GEMINI_FLASH
from my_keys import GEMINI_API_KEY
from my_helper import encode_image
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.globals import set_debug

set_debug(True)

llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_FLASH)

imagen = encode_image("datos/ejemplo_grafico.jpg")

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
                    "image_url": {"url": "data:image/jpeg;base64,{imagen_informada}"},
                },
            ],
        ),
    ]
)

parser_json = JsonOutputParser(pydantic_object=DetallesImagen)

cadena_analisis = template_analisis | llm | StrOutputParser()

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

# CORRECCIÓN: Se inyecta el resultado de análisis explícitamente en la variable que espera el resumen
cadena_compuesta = {"respuesta_analisis_imagen": cadena_analisis} | cadena_resumen

if __name__ == "__main__":
    respuesta = cadena_compuesta.invoke({"imagen_informada": imagen})
    print(respuesta)
