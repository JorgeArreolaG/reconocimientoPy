from deepface import DeepFace

# Rutas de las imágenes
persona_conocida = "images/persona1.jpg"
imagen_prueba = "images/test1.jpg"

# Verificar si son la misma persona
resultado = DeepFace.verify(persona_conocida, imagen_prueba)
print("¿Es la misma persona?:", resultado["verified"])
