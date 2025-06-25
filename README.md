# Análisis de Redes Sociales con Grafos

Este proyecto implementa un sistema para cargar, construir y analizar redes sociales representadas como grafos, utilizando datos de ubicaciones geográficas y conexiones entre usuarios. Utiliza bibliotecas como `networkx`, `geopandas`, `polars`, y `plotly` para procesar grandes conjuntos de datos y generar visualizaciones interactivas.

## Descripción

El código principal, ubicado en `2.py`, permite:
- Cargar datos de ubicaciones (`10_million_location.txt`) y conexiones de usuarios (`10_million_user.txt`).
- Construir un grafo (`networkx.Graph`) con nodos (usuarios) y aristas (conexiones).
- Filtrar ubicaciones geográficas para asegurar que estén en tierra firme usando `geopandas`.
- Analizar propiedades de la red, como densidad, grado promedio, y comunidades.
- Generar visualizaciones interactivas de la red y ubicaciones en un mapa mundial.

El proyecto está diseñado para manejar grandes volúmenes de datos de manera eficiente, con monitoreo de uso de memoria y procesamiento por lotes.

## Requisitos

- Python 3.8 o superior
- Dependencias:
  ```bash
  pip install polars networkx geopandas plotly psutil tqdm numpy shapely community matplotlib
  ```
- Archivos de datos:
  - `10_million_location.txt`: Contiene latitudes y longitudes de usuarios, formato CSV sin encabezado.
  - `10_million_user.txt`: Contiene listas de conexiones de usuarios, formato CSV.
  - Shapefile de países: `ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp` (descargar desde [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)).

## Instalación

1. Clona el repositorio o copia el código en tu máquina.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Asegúrate de que los archivos de datos (`10_million_location.txt`, `10_million_user.txt`) y el shapefile estén en el directorio correcto.

## Uso

Ejecuta el script principal:
```bash
python 2.py
```

El script:
- Carga los datos de ubicaciones y usuarios (limitado a 1,000 nodos por defecto).
- Construye un grafo con las conexiones válidas.
- Analiza propiedades de la red (densidad, comunidades, caminos más cortos, etc.).
- Genera dos visualizaciones HTML:
  - `network_map_visualization_1000.html`: Red de conexiones sobre un mapa mundial.
  - `map_visualization.html`: Mapa de ubicaciones de usuarios.

### Configuración

Puedes ajustar parámetros en la función `main()`:
```python
MAX_NODOS = 1_000  # Número máximo de nodos a procesar
VIS_NODOS = 1_000  # Número de nodos para la visualización
```

## Estructura del Código

El código está organizado en dos clases principales:

### `GraphDataLoader`
Responsable de cargar datos, construir el grafo y generar visualizaciones.

**Fragmento clave: Construcción del grafo**
```python
def construir_grafo(self, max_nodos: int = 100_000) -> nx.Graph:
    valid_nodes = set(self.locations_df['user_id'].to_list())
    G = nx.Graph()
    print(f"🔧 Construyendo grafo (máximo {max_nodos:,} nodos)...")
    for i, conexiones in enumerate(tqdm(usuarios_a_procesar, desc="Construyendo grafo")):
        user_id = i + 1
        if user_id not in valid_nodes:
            continue
        for conn in conexiones:
            if user_id != conn and conn <= max_nodos and conn in valid_nodes:
                G.add_edge(user_id, conn)
    self.graph = G
    return G
```

### `GraphAnalyzer`
Analiza propiedades de la red, como comunidades, caminos más cortos, y árboles de expansión mínima.

**Fragmento clave: Detección de comunidades**
```python
def detect_communities_louvain(self) -> dict:
    if isinstance(self.graph, nx.DiGraph):
        G_undirected = self.graph.to_undirected()
    else:
        G_undirected = self.graph
    return community.best_partition(G_undirected)
```

## Ejemplo de Salida

La ejecución genera logs detallados y archivos HTML. Ejemplo de log:
```
2025-06-25 11:30:23,510 - graph_loader - INFO - Grafo construido exitosamente:
  - Nodos: 1,000
  - Aristas: 5,834
  - Densidad: 0.011680
2025-06-25 11:30:30,787 - graph_loader - INFO - Comunidades detectadas (Louvain): 15
2025-06-25 11:30:31,219 - graph_loader - INFO - Longitud promedio del camino más corto: 3.20
```

Archivos generados:
- `network_map_visualization_1000.html`: Visualización interactiva de la red.
- `map_visualization.html`: Mapa con ubicaciones de usuarios.

## Visualizaciones

Las visualizaciones usan `plotly` para mostrar:
- **Red de conexiones**: Nodos (usuarios) y aristas (conexiones) en un mapa mundial, con colores según el grado de conexión.
- **Mapa de ubicaciones**: Puntos geográficos de usuarios.

**Fragmento clave: Visualización de la red**
```python
fig.add_trace(go.Scattergeo(
    lat=edge_lats,
    lon=edge_lons,
    mode='lines',
    line=dict(width=1, color='rgba(255, 0, 0, 0.3)'),
    name='Conexiones'
))
```

## Contribuciones

Si deseas contribuir:
1. Crea un fork del repositorio.
2. Implementa tus cambios en una rama nueva.
3. Envía un pull request con una descripción clara de los cambios.

## Licencia

Este proyecto está bajo la licencia MIT. Consulta el archivo `LICENSE` para más detalles.