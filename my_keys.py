import os
from dotenv import load_dotenv

# Esto busca el archivo .env y carga las variables al sistema
load_dotenv()

# Asignamos los valores a las variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# TRUCO: LangChain busca por defecto la variable "GOOGLE_API_KEY". 
# Es una excelente práctica igualarla aquí mismo para evitar errores futuros:
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY