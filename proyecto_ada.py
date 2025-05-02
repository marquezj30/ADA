import pandas as pd
from tqdm import tqdm


# Leer ubicaciones con pandas

def leer_ubicaciones(path):
    try:
        print(" Leyendo archivo de ubicaciones con pandas...")
        ubicaciones = pd.read_csv(path, header=None, names=["latitud", "longitud"])
        print(ubicaciones.head())
        print(f" Se leyeron {len(ubicaciones):,} ubicaciones.\n")
        return ubicaciones
    except Exception as e:
        print(f" Error al leer ubicaciones: {e}")
        return None


# Leer usuarios con tqdm y open

def leer_usuarios(path):
    try:
        print("Leyendo archivo de usuarios línea por línea...")
        usuarios = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, linea in enumerate(tqdm(f, total=10_000_000)):
                conexiones = list(map(int, linea.strip().split(',')))
                usuarios.append(conexiones)
        print(f"Se leyeron {len(usuarios):,} listas de usuarios.\n")
        return usuarios
    except Exception as e:
        print(f"Error al leer usuarios: {e}")
        return None


# Consultar un usuario

def consultar_usuario(usuario_id, ubicaciones, usuarios):
    try:
        # Verificar si el id de usuario es válido
        if usuario_id < 1 or usuario_id > len(ubicaciones):
            print("ID de usuario no válido.")
            return
        
        # Obtener ubicación y conexiones del usuario
        ubicacion = ubicaciones.iloc[usuario_id - 1]
        conexiones = usuarios[usuario_id - 1]
        
        print(f"\nConsulta para Usuario {usuario_id}:")
        print(f"Ubicación: Latitud: {ubicacion['latitud']}, Longitud: {ubicacion['longitud']}")
        print(f"Conexiones (sigue a {len(conexiones)} usuarios): {conexiones[:10]} ...")  # Muestra las primeras 10 conexiones
        
    except Exception as e:
        print(f"Error al consultar usuario: {e}")


# Rutas a los archivos descomprimidos

archivo_ubicaciones = "10_million_location.txt"
archivo_usuarios = "10_million_user.txt"


# Ejecutar funciones

if __name__ == "__main__":
    # Leer los archivos
    ubicaciones = leer_ubicaciones(archivo_ubicaciones)
    usuarios = leer_usuarios(archivo_usuarios)

    if ubicaciones is not None and usuarios is not None:
        # Verificación de integridad
        if len(ubicaciones) == 10_000_000 and len(usuarios) == 10_000_000:
            print("Ambos archivos tienen exactamente 10 millones de líneas. Todo correcto.\n")
            
            # Consultar un usuario específico
            while True:
                try:
                    usuario_id = int(input("Ingrese el ID de usuario para consultar (1 a 10,000,000) o 0 para salir: "))
                    if usuario_id == 0:
                        break
                    consultar_usuario(usuario_id, ubicaciones, usuarios)
                except ValueError:
                    print("Por favor, ingrese un número válido.")
        else:
            print("Verificación incompleta de archivos.")
    else:
        print("Error al leer los archivos.")