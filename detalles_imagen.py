from pydantic import BaseModel, Field
from typing import List

class DetallesImagen(BaseModel):
    titulo: str = Field(..., description="Define un título breve y descriptivo para la imagen")
    descripcion: str = Field(..., description="coloca aqui una Descripción de la imagen")
    etiquetas: List[str] = Field(..., description="Define una lista con 3 palabras clave separadas por comas")