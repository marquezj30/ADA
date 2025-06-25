# Análisis de Redes Sociales con Grafos

Este proyecto implementa un sistema para cargar, construir y analizar redes sociales representadas como grafos, utilizando datos de ubicaciones geográficas y conexiones entre usuarios. Utiliza bibliotecas como `networkx`, `geopandas`, `polars`, y `plotly` para procesar grandes conjuntos de datos y generar visualizaciones interactivas.

## Descripción

El código principal, ubicado en `poryecto_ada.py`, permite:
- Cargar datos de ubicaciones (`10_million_location.txt`) y conexiones de usuarios (`10_million_user.txt`).
- Construir un grafo (`networkx.Graph`) con nodos (usuarios) y aristas (conexiones).
- Filtrar ubicaciones geográficas para asegurar que estén en tierra firme usando `geopandas`.
- Analizar propiedades de la red, como densidad, grado promedio, y comunidades.
- Generar visualizaciones interactivas de la red y ubicaciones en un mapa mundial.

El proyecto está diseñado para manejar grandes volúmenes de datos de manera eficiente, con monitoreo de uso de memoria y procesamiento por lotes.

## Requisitos

**Bibliotecas Utilizadas en el Proyecto de Análisis de Redes Sociales**

```pip install polars networkx geopandas plotly psutil tqdm numpy shapely python-louvain```

 las bibliotecas de Python utilizadas en el proyecto para cargar, procesar, analizar y visualizar datos de una red social representada como un grafo. Cada biblioteca se describe con su propósito y un ejemplo de uso en el código.

**1. `os`**
**Propósito**: Proporciona funciones para interactuar con el sistema operativo, como obtener el ID del proceso actual para monitorear el uso de memoria. En el proyecto, se usa para calcular el consumo de memoria, reportado como 1325.25 MB tras construir el grafo.

**Ejemplo de uso**:
```python
def _memory_usage(self) -> str:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"{mem_info.rss / 1024 / 1024:.2f} MB"
```

**2. `time`**
**Propósito**: Mide el tiempo de ejecución del programa. En el proyecto, se usa para registrar el tiempo total de procesamiento, que fue de 165.74 segundos según la salida.

**Ejemplo de uso**:
```python
def main():
    start_time = time.time()
    # ... (código del análisis)
    logger.info(f"Proceso finalizado en {time.time() - start_time:.2f}s")
```

**3. `logging`**
**Propósito**: Registra mensajes informativos, advertencias y errores en un archivo (`graph_loader.log`) y en la consola. Se utiliza para rastrear el progreso, como la construcción del grafo (1,000 nodos, 5,834 aristas) y la detección de comunidades.

**Ejemplo de uso**:
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("graph_loader")
```

**4. `pathlib`**
**Propósito**: Maneja rutas de archivos de forma robusta y multiplataforma. En el proyecto, valida la existencia de los archivos de entrada (`10_million_location.txt` y `10_million_user.txt`).

**Ejemplo de uso**:
```python
def _validate_input_files(self) -> None:
    if not self.location_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {self.location_path}")
    if not self.user_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {self.user_path}")
```

**5. `gc`**
**Propósito**: Controla el recolector de basura para liberar memoria manualmente. Se usa en el procesamiento por lotes de la lista de adyacencia para optimizar el uso de memoria.

**Ejemplo de uso**:
```python
def process_adjacency_list(self) -> int:
    # ... (procesamiento por lotes)
    del batch_data
    gc.collect()
```

**6. `psutil`**
**Propósito**: Monitorea recursos del sistema, como el uso de memoria del proceso. En el proyecto, calcula el consumo de memoria, reportado como 1325.25 MB tras construir el grafo.

**Ejemplo de uso**:
```python
def _memory_usage(self) -> str:
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"{mem_info.rss / 1024 / 1024:.2f} MB"
```

**7. `heapq`**
**Propósito**: Implementa colas de prioridad basadas en montones, usadas en el algoritmo de Dijkstra para encontrar caminos más cortos. En la salida, Dijkstra calculó un camino de distancia 3 entre los nodos 1 y 500.

**Ejemplo de uso**:
```python
def find_shortest_path_dijkstra(self, start: int, end: int) -> tuple:
    heap = [(0, start)]
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        # ... (lógica de Dijkstra)
        heapq.heappush(heap, (distance, neighbor))
```

**8. `polars`**
**Propósito**: Proporciona un marco eficiente para manipulación de datos tabulares. En el proyecto, carga y filtra las ubicaciones geográficas, procesando 8,563,494 ubicaciones válidas sobre tierra.

**Ejemplo de uso**:
```python
def load_locations(self) -> pl.DataFrame:
    df = pl.read_csv(self.location_path, has_header=False, separator=",",
                     new_columns=["latitude", "longitude"])
    df = df.with_row_index(name="user_id", offset=1)
```

**9. `networkx`**
**Propósito**: Crea, manipula y analiza grafos. Es central en el proyecto, usada para construir el grafo (1,000 nodos, 5,834 aristas), calcular la densidad (0.01168) y ejecutar algoritmos como PageRank.

**Ejemplo de uso**:
```python
def construir_grafo(self, max_nodos: int = 100_000) -> nx.Graph:
    G = nx.Graph()
    for i, conexiones in enumerate(tqdm(usuarios_a_procesar)):
        for conn in conexiones:
            if user_id != conn and conn <= max_nodos and conn in valid_nodes:
                G.add_edge(user_id, conn)
    return G
```

**10. `geopandas`**
**Propósito**: Maneja datos geoespaciales, usada para filtrar ubicaciones que estén en tierra firme usando un shapefile de países. En la salida, se filtraron 8,563,494 ubicaciones válidas.

**Ejemplo de uso**:
```python
def load_locations(self) -> pl.DataFrame:
    gdf = gpd.GeoDataFrame(df.to_pandas(),
                           geometry=[Point(xy) for xy in zip(df["longitude"].to_list(), df["latitude"].to_list())],
                           crs="EPSG:4326")
    world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
    land = world.geometry.union_all()
    filtered = gdf[gdf.geometry.within(land)]
```

**11. `plotly.express` y `plotly.graph_objects`**
**Propósito**: Crean visualizaciones interactivas. `plotly.express` genera el mapa de ubicaciones (`map_visualization.html`), y `plotly.graph_objects` visualiza la red de conexiones (`network_map_visualization_1000.html`) con 5,834 conexiones.

**Ejemplo de uso**:
```python
def plot_mapbox(self, sample_size: int = 1000) -> None:
    df = self.locations_df.head(sample_size).to_pandas()
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="user_id",
        title=f"Ubicaciones de {sample_size} usuarios"
    )
    fig.write_html("map_visualization.html")
```

**12. `community` (`python-louvain`)**
**Propósito**: Implementa el algoritmo de Louvain para detectar comunidades maximizando la modularidad. En la salida, se identificaron 15 comunidades en el grafo.

**Ejemplo de uso**:
```python
def detect_communities_louvain(self) -> dict:
    G_undirected = self.graph.to_undirected() if isinstance(self.graph, nx.DiGraph) else self.graph
    return community.best_partition(G_undirected)
```

**13. `shapely.geometry`**
**Propósito**: Proporciona objetos geométricos como `Point` para operaciones geoespaciales, usado con `geopandas` para crear geometrías de puntos a partir de coordenadas.

**Ejemplo de uso**:
```python
def load_locations(self) -> pl.DataFrame:
    gdf = gpd.GeoDataFrame(df.to_pandas(),
                           geometry=[Point(xy) for xy in zip(df["longitude"].to_list(), df["latitude"].to_list())],
                           crs="EPSG:4326")
```

**14. `numpy`**
**Propósito**: Realiza cálculos numéricos, como el grado promedio (11.668) y el grado máximo (215) en el análisis de propiedades del grafo.

**Ejemplo de uso**:
```python
def analyze_network_properties(self) -> dict:
    degrees = [d for n, d in self.graph.degree()]
    properties['avg_degree'] = np.mean(degrees)
```

**15. `tqdm`**
**Propósito**: Muestra barras de progreso para iteraciones largas, como la construcción del grafo (procesado a 6,614.55 iteraciones por segundo en la salida).

**Ejemplo de uso**:
```python
def construir_grafo(self, max_nodos: int = 100_000) -> nx.Graph:
    for i, conexiones in enumerate(tqdm(usuarios_a_procesar, desc="Construyendo grafo")):
        # ... (lógica de construcción)
```

**16. `random`**
**Propósito**: Genera números aleatorios para barajar nodos en `label_propagation` y muestrear pares en `average_shortest_path`. En la salida, ayudó a calcular la longitud promedio del camino (3.20).

**Ejemplo de uso**:
```python
def label_propagation(self, max_iter=1000) -> list:
    nodes = list(self.graph_dict.keys())
    for i in range(max_iter):
        random.shuffle(nodes)
```

**17. `math`**
**Propósito**: Proporciona funciones matemáticas como `sin` y `atan2` para la fórmula de Haversine en `kruskal_mst`, que calculó un árbol de expansión mínima con un peso total de 15,013.33 km.

**Ejemplo de uso**:
```python
def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    dlat = math.radians(lat2 - lat1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
```

**18. `collections` (`deque` y `Counter`)**
**Propósito**: Proporciona estructuras de datos especializadas. `deque` se usa en `bfs_shortest_path` para búsqueda en anchura, y `Counter` en `label_propagation` para contar etiquetas de vecinos, detectando 12 comunidades en la salida.

**Ejemplo de uso**:
```python
def label_propagation(self, max_iter=1000) -> list:
    neighbor_labels = [labels[neighbor] for neighbor in self.graph_dict[node] if neighbor in labels]
    if neighbor_labels:
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
```
## Uso

Ejecuta el script principal:
```bash
python proyecto_ada.py
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

**GraphDataLoader**

La función  crea un grafo no dirigido (nx.Graph) a partir de los datos de conexiones de usuarios, asegurándose de que los nodos tengan ubicaciones geográficas válidas. Filtra conexiones para incluir solo nodos dentro del límite especificado (max_nodos) y presentes en el conjunto de nodos válidos. La salida muestra un grafo con 1,000 nodos y 5,834 aristas, con una densidad de 0.01168.

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

**Análisis de Propiedades de la Red**

Esta función calcula propiedades fundamentales del grafo, como el número de nodos, aristas, densidad, grado promedio, grado máximo, conectividad y los nodos más influyentes según PageRank. En la salida, se reporta un grado promedio de 11.668, un grado máximo de 215, y que el grafo no es completamente conectado (5 componentes conectadas). También identifica los 5 nodos más influyentes, liderados por el usuario 488 con un puntaje PageRank de 0.0191.

```python
def analyze_network_properties(self) -> dict:
    properties = {}
    properties['num_nodes'] = self.graph.number_of_nodes()
    properties['num_edges'] = self.graph.number_of_edges()
    properties['density'] = nx.density(self.graph)
    degrees = [d for n, d in self.graph.degree()]
    properties['avg_degree'] = np.mean(degrees)
    properties['max_degree'] = max(degrees) if degrees else 0
    properties['is_connected'] = nx.is_connected(self.graph)
    properties['num_connected_components'] = nx.number_connected_components(self.graph)
    pagerank = nx.pagerank(self.graph)
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
    properties['top_influential_nodes'] = top_pagerank
    return properties
```
- **Red de conexiones**: Nodos (usuarios) y aristas (conexiones) en un mapa mundial, con colores según el grado de conexión.
- **Mapa de ubicaciones**: Puntos geográficos de usuarios.

**Visualización de la red**
```python
fig.add_trace(go.Scattergeo(
    lat=edge_lats,
    lon=edge_lons,
    mode='lines',
    line=dict(width=1, color='rgba(255, 0, 0, 0.3)'),
    name='Conexiones'
))
```
**Detección de Comunidades con Louvain**
El método detect_communities_louvain utiliza el algoritmo de Louvain para identificar comunidades en el grafo, maximizando la modularidad. Este algoritmo agrupa nodos en comunidades basándose en la densidad de conexiones internas frente a conexiones externas. Si el grafo es dirigido, se convierte a no dirigido para el análisis.

```python
def detect_communities_louvain(self) -> dict:
    if isinstance(self.graph, nx.DiGraph):
        G_undirected = self.graph.to_undirected()
    else:
        G_undirected = self.graph
    return community.best_partition(G_undirected)

```
**Explicacion**: Este método toma el grafo y, si es necesario, lo convierte a un grafo no dirigido. Luego, aplica el algoritmo de Louvain (implementado en la biblioteca community) para asignar a cada nodo una comunidad, devolviendo un diccionario donde las claves son los nodos y los valores son los identificadores de las comunidades. En la salida de ejemplo, se detectaron 15 comunidades en un grafo de 1,000 nodos.

**Detección de Comunidades con Propagación de Etiquetas(louvein a mano)**

```python
def label_propagation(self, max_iter=1000) -> list:
    labels = {node: node for node in self.graph_dict}
    nodes = list(self.graph_dict.keys())
    for i in range(max_iter):
        random.shuffle(nodes)
        changed = False
        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in self.graph_dict[node] if neighbor in labels]
            if neighbor_labels:
                most_common = Counter(neighbor_labels).most_common(1)[0][0]
                if labels[node] != most_common:
                    labels[node] = most_common
                    changed = True
        if not changed:
            logger.info(f"Convergencia alcanzada en iteración {i+1}")
            break
    communities = {}
    for node, label in labels.items():
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    return list(communities.values())
```
**Explicación:** Implementa un algoritmo de propagación de etiquetas para detectar comunidades, donde cada nodo adopta la etiqueta más común entre sus vecinos. El proceso itera hasta la convergencia, que en la salida se alcanzó en la iteración 5, detectando 12 comunidades, con la comunidad más grande conteniendo 962 nodos.

**Cálculo de la Longitud Promedio del Camino Más Corto**

```python
def average_shortest_path(self, num_samples: int = 10000) -> float:
    nodes = list(self.graph_dict.keys())
    total_dist = 0
    count = 0
    for _ in range(num_samples):
        start, end = random.sample(nodes, 2)
        dist = self.bfs_shortest_path(start, end)
        if dist < float('inf'):
            total_dist += dist
            count += 1
    avg = total_dist / count if count > 0 else float('inf')
    logger.info(f"Longitud promedio del camino más corto: {avg:.2f} (basado en {count} caminos válidos)")
    return avg
```
**Explicación:** Estima la longitud promedio del camino más corto entre pares de nodos usando muestreo aleatorio y búsqueda en anchura (BFS). En la salida, se calculó con 1,000 muestras, obteniendo una longitud promedio de 3.20 basada en 986 caminos válidos.

**Algoritmo de Kruskal para Árbol de Expansión Mínima**

El método kruskal_mst implementa el algoritmo de Kruskal para construir un árbol de expansión mínima (MST) basado en las distancias geográficas entre nodos, calculadas con la fórmula de Haversine. Este árbol conecta todos los nodos con el menor peso total posible, utilizando las coordenadas de latitud y longitud.

```python
def kruskal_mst(self) -> list:
    edges = []
    skipped_edges = 0
    for u in self.graph_dict:
        if u not in self.locations:
            skipped_edges += len(self.graph_dict[u])
            continue
        lat1, lon1 = self.locations[u]
        for v in self.graph_dict[u]:
            if u < v and v in self.locations:
                lat2, lon2 = self.locations[v]
                weight = self.haversine(lat1, lon1, lat2, lon2)
                edges.append((u, v, weight))
            else:
                skipped_edges += 1
    edges = sorted(edges, key=lambda x: x[2])
    nodes = list(self.graph_dict.keys())
    uf = self.UnionFind(nodes)
    mst = []
    total_weight = 0
    for u, v, weight in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            mst.append((u, v, weight))
            total_weight += weight
    logger.info(f"Árbol de expansión mínima construido con {len(mst)} aristas y peso total {total_weight:.2f} km")
    return mst
```
**Explicacion** Este método calcula las distancias geográficas entre nodos conectados usando la fórmula de Haversine y ordena las aristas por peso. Utiliza una estructura UnionFind para evitar ciclos y construir el MST seleccionando las aristas de menor peso que conecten componentes distintas. En la salida de ejemplo, el MST tiene 995 aristas con un peso total de 15,013.33 km, y se omitieron 5,834 aristas debido a nodos sin ubicación.

**Algoritmo de Dijkstra para Caminos Más Cortos**
El método find_shortest_path_dijkstra implementa el algoritmo de Dijkstra para encontrar el camino más corto entre dos nodos en el grafo, considerando pesos uniformes (peso 1 para cada arista). Devuelve la distancia del camino y la lista de nodos que lo componen.

```python
def find_shortest_path_dijkstra(self, start: int, end: int) -> tuple:
    visited = set()
    distances = {node: float('inf') for node in self.graph.nodes()}
    previous = {node: None for node in self.graph.nodes()}
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        current_distance, current_node = heapq.heappop(heap)
        if current_node in visited:
            continue
        visited.add(current_node)
        for neighbor in self.graph.neighbors(current_node):
            weight = 1
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(heap, (distance, neighbor))
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return distances[end], path
```
**Explicacion** Este método utiliza una cola de prioridad (heapq) para explorar los nodos del grafo, actualizando las distancias más cortas desde el nodo inicial (start) hasta los demás. Cuando se alcanza el nodo final (end), reconstruye el camino siguiendo los nodos previos. En la salida de ejemplo, el camino más corto entre los nodos 1 y 500 tiene una distancia de 3 y pasa por los nodos [1, 775, 205, 500].

**Visualización de Ubicaciones en Mapa**

```python
def plot_mapbox(self, sample_size: int = 1000) -> None:
    df = self.locations_df.head(sample_size).to_pandas()
    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="user_id",
        height=700,
        title=f"Ubicaciones de {sample_size} usuarios"
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.write_html("map_visualization.html")
```
**Explicación:** Crea un mapa interactivo con las ubicaciones geográficas de los usuarios usando plotly. Cada punto representa un usuario en sus coordenadas de latitud y longitud. La salida indica que se guardó como map_visualization.html.


**Ejemplo de Salida**


```
C:\Users\user\OneDrive\Escritorio\NUEVO_ADA>python proyecto_ada.py
2025-06-25 11:27:50,121 - graph_loader - INFO - Archivos encontrados: 10_million_location.txt, 10_million_user.txt
2025-06-25 11:27:50,122 - graph_loader - INFO - Iniciando carga de ubicaciones. Memoria antes: 118.93 MB
2025-06-25 11:27:50,349 - graph_loader - INFO - Filtrando ubicaciones geográficamente...
2025-06-25 11:30:21,214 - graph_loader - INFO - Ubicaciones válidas sobre tierra: 8563494
2025-06-25 11:30:22,490 - graph_loader - INFO - Cargando datos de usuarios. Memoria inicial: 718.86 MB
2025-06-25 11:30:22,944 - graph_loader - INFO - Datos de 1,000 usuarios cargados en memoria
🔧 Construyendo grafo (máximo 1,000 nodos)...
Construyendo grafo: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 6614.55it/s]
Grafo construido con 1,000 nodos y 5,834 aristas.

2025-06-25 11:30:23,510 - graph_loader - INFO - Grafo construido exitosamente:
2025-06-25 11:30:23,510 - graph_loader - INFO -   - Nodos: 1,000
2025-06-25 11:30:23,511 - graph_loader - INFO -   - Aristas: 5,834
2025-06-25 11:30:23,511 - graph_loader - INFO -   - Densidad: 0.011680
2025-06-25 11:30:23,511 - graph_loader - INFO -   - Memoria: 1325.25 MB
2025-06-25 11:30:30,393 - graph_loader - INFO - Analizando propiedades de la red...
2025-06-25 11:30:30,562 - graph_loader - INFO - Propiedades de la red:
2025-06-25 11:30:30,562 - graph_loader - INFO -   num_nodes: 1000
2025-06-25 11:30:30,563 - graph_loader - INFO -   num_edges: 5834
2025-06-25 11:30:30,564 - graph_loader - INFO -   density: 0.011679679679679679
2025-06-25 11:30:30,567 - graph_loader - INFO -   avg_degree: 11.668
2025-06-25 11:30:30,568 - graph_loader - INFO -   max_degree: 215
2025-06-25 11:30:30,568 - graph_loader - INFO -   is_connected: False
2025-06-25 11:30:30,568 - graph_loader - INFO -   num_connected_components: 5
2025-06-25 11:30:30,569 - graph_loader - INFO - Top 5 nodos más influyentes (PageRank):
2025-06-25 11:30:30,569 - graph_loader - INFO -   Usuario 488: 0.0191
2025-06-25 11:30:30,569 - graph_loader - INFO -   Usuario 24: 0.0148
2025-06-25 11:30:30,569 - graph_loader - INFO -   Usuario 290: 0.0124
2025-06-25 11:30:30,570 - graph_loader - INFO -   Usuario 12: 0.0119
2025-06-25 11:30:30,570 - graph_loader - INFO -   Usuario 1: 0.0099
2025-06-25 11:30:30,787 - graph_loader - INFO - Comunidades detectadas (Louvain): 15
2025-06-25 11:30:30,787 - graph_loader - INFO - Iniciando detección de comunidades con propagación de etiquetas
2025-06-25 11:30:30,808 - graph_loader - INFO - Convergencia alcanzada en iteración 5
2025-06-25 11:30:30,808 - graph_loader - INFO - Se detectaron 12 comunidades
2025-06-25 11:30:30,809 - graph_loader - INFO - Comunidades detectadas (Propagación de etiquetas): 12
2025-06-25 11:30:30,809 - graph_loader - INFO -   Comunidad 1: 962 nodos (primeros 5: [1, 2, 3, 4, 5])
2025-06-25 11:30:30,811 - graph_loader - INFO -   Comunidad 2: 2 nodos (primeros 5: [65, 66])
2025-06-25 11:30:30,813 - graph_loader - INFO -   Comunidad 3: 3 nodos (primeros 5: [128, 130, 131])
2025-06-25 11:30:30,814 - graph_loader - INFO -   Comunidad 4: 3 nodos (primeros 5: [135, 136, 137])
2025-06-25 11:30:30,815 - graph_loader - INFO -   Comunidad 5: 3 nodos (primeros 5: [140, 141, 142])
2025-06-25 11:30:30,815 - graph_loader - INFO - Calculando longitud promedio del camino más corto con 1000 muestras
2025-06-25 11:30:31,219 - graph_loader - INFO - Longitud promedio del camino más corto: 3.20 (basado en 986 caminos válidos)
2025-06-25 11:30:31,219 - graph_loader - INFO - Longitud promedio del camino más corto: 3.20
2025-06-25 11:30:31,223 - graph_loader - INFO - Calculando árbol de expansión mínima con Kruskal
2025-06-25 11:30:31,233 - graph_loader - WARNING - Se omitieron 5834 aristas debido a nodos sin ubicación
2025-06-25 11:30:31,237 - graph_loader - INFO - Árbol de expansión mínima construido con 995 aristas y peso total 15013.33 km
2025-06-25 11:30:31,238 - graph_loader - INFO - Árbol de expansión mínima: 995 aristas
2025-06-25 11:30:31,238 - graph_loader - INFO -   Arista: 488 -> 608, peso: 0.26 km
2025-06-25 11:30:31,239 - graph_loader - INFO -   Arista: 446 -> 458, peso: 0.38 km
2025-06-25 11:30:31,240 - graph_loader - INFO -   Arista: 25 -> 270, peso: 0.54 km
2025-06-25 11:30:31,242 - graph_loader - INFO -   Arista: 516 -> 535, peso: 0.56 km
2025-06-25 11:30:31,245 - graph_loader - INFO -   Arista: 488 -> 661, peso: 0.58 km
2025-06-25 11:30:31,248 - graph_loader - INFO - Dijkstra 1 -> 500: distancia = 3, camino = [1, 775, 205, 500]...
2025-06-25 11:30:31,248 - graph_loader - INFO - Generando visualización con 1000 nodos...
2025-06-25 11:30:31,249 - graph_loader - INFO - Visualizando red con conexiones para 1000 nodos sobre mapa mundi
2025-06-25 11:30:35,365 - graph_loader - INFO - Visualización de red en mapa guardada como network_map_visualization_1000.html
2025-06-25 11:30:35,366 - graph_loader - INFO - Estadísticas del grafo:
2025-06-25 11:30:35,368 - graph_loader - INFO - - Nodos: 1000
2025-06-25 11:30:35,369 - graph_loader - INFO - - Aristas: 5834
2025-06-25 11:30:35,370 - graph_loader - INFO - - Densidad: 0.0117
2025-06-25 11:30:35,370 - graph_loader - INFO - - Conexiones mostradas en mapa: 5834
2025-06-25 11:30:35,371 - graph_loader - INFO - Visualizando ubicaciones en mapa mundi con Plotly
2025-06-25 11:30:35,863 - graph_loader - INFO - Mapa guardado como map_visualization.html
2025-06-25 11:30:35,864 - graph_loader - INFO - Proceso finalizado en 165.74s
2025-06-25 11:30:35,865 - graph_loader - INFO - Archivos generados:
2025-06-25 11:30:35,866 - graph_loader - INFO -   - network_map_visualization_1000.html (Red con conexiones sobre mapa mundi)
2025-06-25 11:30:35,866 - graph_loader - INFO -   - map_visualization.html (Mapa de ubicaciones)
```
Archivos generados:
- `network_map_visualization_1000.html`: Visualización interactiva de la red.
- `map_visualization.html`: Mapa con ubicaciones de usuarios.
