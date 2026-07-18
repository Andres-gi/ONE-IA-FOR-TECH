from orquestador import AgenteOrquestador


def main():
    # 1. Instanciamos el orquestador (que ya tiene el AgentExecutor configurado por dentro)
    orquestador = AgenteOrquestador()

    # 2. Definimos el prompt o instrucción para el agente
    pregunta = (
        "1. Analiza la imagen ubicada en datos/ejemplo_grafico.jpg. "
        "2. Luego, toma las etiquetas que encontraste y utiliza tu herramienta "
        "de explicación para explicarme detalladamente qué significan."
    )

    print(f"Enviando solicitud: '{pregunta}'\n")

    # 3. Utilizamos el método 'ejecutar' que ya programamos en la clase
    respuesta = orquestador.ejecutar(pregunta)

    # 4. Imprimimos el resultado final
    print("\n--- RESPUESTA FINAL ---")
    print(respuesta)


if __name__ == "__main__":
    main()
