#  Proyecto: Carga Masiva de Usuarios y Ubicaciones 

**Grupo Bayern**  
**Integrantes**:
- Johan Márquez Zúñiga  
- Marcelo Silva Cabrera  

---

##  Descripción

Este proyecto en Python permite la lectura de archivos masivos de datos con **10 millones de registros:  
- `10_million_location.txt` (ubicaciones geográficas)  
- `10_million_user.txt` (conexiones entre usuarios)

El sistema permite consultar cualquier usuario e imprimir su ubicación y las conexiones con otros usuarios.

---

##  Funcionalidades

- **Carga Masiva Eficiente**: Utiliza `pandas` para leer el archivo de ubicaciones y `tqdm` para leer usuarios línea por línea. Ambos enfoques permiten manejar grandes volúmenes de datos de forma eficiente.
  
- **Manejo de Errores y Logs**: Implementación robusta de manejo de errores y logs detallados que registran cada operación realizada (por ejemplo, lectura de archivos y consultas de usuarios).
  
- **Consulta de Usuarios**: Permite consultar un usuario por su ID, obteniendo su ubicación geográfica y un resumen de sus conexiones.

---

##  Código

### 1. Lectura de Ubicaciones con `pandas`

Primero, el código usa la librería `pandas` para leer el archivo de ubicaciones (`10_million_location.txt`). Este archivo tiene dos columnas: latitud y longitud, las cuales se asignan como encabezados de columna.


import pandas as pd

# -----------------------------
# Leer ubicaciones con pandas
# -----------------------------
#def leer_ubicaciones(path):
    try:
        print("📍 Leyendo archivo de ubicaciones con pandas...")
        ubicaciones = pd.read_csv(path, header=None, names=["latitud", "longitud"])
        print(ubicaciones.head())
        print(f"✅ Se leyeron {len(ubicaciones):,} ubicaciones.\n")
        return ubicaciones
    except Exception as e:
        print(f"❌ Error al leer ubicaciones: {e}")
        return None


**Función leer_ubicaciones(path)**: Esta función recibe la ruta al archivo de ubicaciones como parámetro. Usa pd.read_csv() para leer el archivo txt. Se le pasa 

header=None para que no se espere una fila de encabezados y names=["latitud", "longitud"] para asignar nombres a las columnas.

**Manejo de Errores**: Si ocurre un error al intentar leer el archivo (por ejemplo, si el archivo no existe o hay un problema con el formato), se captura la excepción y se imprime un mensaje de error.

**Retorno**: Si todo sale bien, la función retorna un DataFrame con las ubicaciones leídas y muestra las primeras filas para verificar la lectura.

### 2. Lectura de Usuarios con `tqdm` y `open()`

**Función `leer_usuarios(path)`**: Esta función recibe la ruta al archivo de usuarios como parámetro. Usa `tqdm` junto con la función `open()` para leer el archivo línea por línea. Cada línea contiene una lista de IDs de usuarios conectados entre sí. Los IDs son convertidos en enteros y agregados a una lista llamada `usuarios`.

**Manejo de Errores**: Si ocurre un error al intentar leer el archivo (por ejemplo, si el archivo no existe o tiene un formato incorrecto), se captura la excepción y se imprime un mensaje de error.

**Retorno**: Si todo sale bien, la función retorna la lista de usuarios con sus conexiones.


from tqdm import tqdm

# -------------------------------
# Leer usuarios con tqdm y open
# -------------------------------
#def leer_usuarios(path):
    try:
        print("👥 Leyendo archivo de usuarios línea por línea...")
        usuarios = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, linea in enumerate(tqdm(f, total=10_000_000)):
                conexiones = list(map(int, linea.strip().split(',')))
                usuarios.append(conexiones)
        print(f"✅ Se leyeron {len(usuarios):,} listas de usuarios.\n")
        return usuarios
    except Exception as e:
        print(f"❌ Error al leer usuarios: {e}")
        return None


### 3. Consulta de Usuario y Ubicación

Con los datos cargados, la siguiente función permite consultar la ubicación de un usuario a partir de su ID. Si se encuentra el usuario, se imprime su ubicación y las conexiones con otros usuarios.

**Función `consultar_usuario(usuario_id, ubicaciones, usuarios)`**: Esta función consulta un usuario en función de su ID. Primero, obtiene la ubicación de ese usuario del DataFrame de ubicaciones usando `iloc[]`. Luego, busca las conexiones del usuario en la lista `usuarios` y las muestra.

**Manejo de Errores**: Si el `usuario_id` no existe en el DataFrame o la lista, se captura un `IndexError` y se imprime un mensaje de error indicando que el usuario no fue encontrado.

#def consultar_usuario(usuario_id, ubicaciones, usuarios):
    try:
        # Ubicación del usuario
        latitud, longitud = ubicaciones.iloc[usuario_id]
        print(f"🌍 Ubicación del usuario {usuario_id}: Latitud {latitud}, Longitud {longitud}")
        
        # Conexiones del usuario
        conexiones = usuarios[usuario_id]
        print(f"🔗 Conexiones del usuario {usuario_id}: {conexiones[:5]}... (total {len(conexiones)})")
    
    except IndexError:
        print(f"❌ Usuario con ID {usuario_id} no encontrado.")


### 🏁 Ejecución

Para ejecutar este proyecto, simplemente debes llamar las funciones de lectura y consulta. Por ejemplo:



ubicaciones = leer_ubicaciones('10_million_location.txt')
usuarios = leer_usuarios('10_million_user.txt')

# Consulta de un usuario específico
consultar_usuario(12345, ubicaciones, usuarios)

### 🚧 Mejoras Futuras

- **Optimización de Memoria**: Considerar el uso de bases de datos para manejar eficientemente los datos a gran escala.
- **Interfaz de Usuario**: Implementar una interfaz gráfica que permita a los usuarios consultar los datos de manera más interactiva.
- **Multihilo o Multiproceso**: Utilizar procesamiento paralelo para mejorar el rendimiento al leer los archivos grandes.

### 🛠 Requisitos

- Python 3.x
- pandas
- tqdm


