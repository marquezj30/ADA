import os
import time
import logging
from pathlib import Path
import gc
import psutil
import heapq
import polars as pl
import networkx as nx
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import community.community_louvain as community
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm
import random
import math
from collections import deque, Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_loader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("graph_loader")

class GraphDataLoader:
    def __init__(self, location_path: str = "10_million_location.txt", 
                 user_path: str = "10_million_user.txt", 
                 batch_size: int = 500000):
        self.location_path = Path(location_path)
        self.user_path = Path(user_path)
        self.batch_size = batch_size
        self.num_nodes = 0
        self.num_edges = 0
        self.graph = None
        self.locations_df = None
        self.users_data = []
        self._validate_input_files()

    def _validate_input_files(self) -> None:
        if not self.location_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.location_path}")
        if not self.user_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.user_path}")
        logger.info(f"Archivos encontrados: {self.location_path}, {self.user_path}")

    def _memory_usage(self) -> str:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return f"{mem_info.rss / 1024 / 1024:.2f} MB"

    def load_locations(self) -> pl.DataFrame:
        logger.info(f"Iniciando carga de ubicaciones. Memoria antes: {self._memory_usage()}")
        try:
            df = pl.read_csv(self.location_path, has_header=False, separator=",",
                             new_columns=["latitude", "longitude"])
            df = df.with_row_index(name="user_id", offset=1)
            logger.info("Filtrando ubicaciones geográficamente...")
            gdf = gpd.GeoDataFrame(df.to_pandas(),
                                   geometry=[Point(xy) for xy in zip(df["longitude"].to_list(), df["latitude"].to_list())],
                                   crs="EPSG:4326")
            world = gpd.read_file("ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")
            land = world.geometry.union_all()
            filtered = gdf[gdf.geometry.within(land)]
            filtered = pl.from_pandas(filtered.drop(columns="geometry"))
            logger.info(f"Ubicaciones válidas sobre tierra: {len(filtered)}")
            self.num_nodes = len(filtered)
            self.locations_df = filtered
            return filtered
        except Exception as e:
            logger.error(f"Error al cargar ubicaciones con filtro geográfico: {str(e)}")
            raise

    def load_users_data(self, max_users: int = None) -> list:
        logger.info(f"Cargando datos de usuarios. Memoria inicial: {self._memory_usage()}")
        users_data = []
        try:
            with open(self.user_path, 'r') as f:
                user_id = 1
                for line in f:
                    if max_users and user_id > max_users:
                        break
                    line = line.strip()
                    if line:
                        try:
                            connections = [int(x) for x in line.split(',') if x.strip()]
                            users_data.append(connections)
                        except ValueError as e:
                            logger.warning(f"Error en línea {user_id}: {e}")
                            users_data.append([])
                    else:
                        users_data.append([])
                    user_id += 1
                    if user_id % 100000 == 0:
                        logger.info(f"Cargados {user_id:,} usuarios. Memoria: {self._memory_usage()}")
            self.users_data = users_data
            logger.info(f"Datos de {len(users_data):,} usuarios cargados en memoria")
            return users_data
        except Exception as e:
            logger.error(f"Error cargando datos de usuarios: {str(e)}")
            raise

    def construir_grafo(self, max_nodos: int = 100_000) -> nx.Graph:
        if not self.users_data:
            logger.info("Datos de usuarios no cargados. Cargando automáticamente...")
            self.load_users_data(max_nodos)
        if self.locations_df is None:
            logger.info("Ubicaciones no cargadas. Cargando automáticamente...")
            self.load_locations()
        valid_nodes = set(self.locations_df['user_id'].to_list())
        G = nx.Graph()
        print(f"🔧 Construyendo grafo (máximo {max_nodos:,} nodos)...")
        usuarios_a_procesar = self.users_data[:max_nodos]
        for i, conexiones in enumerate(tqdm(usuarios_a_procesar, desc="Construyendo grafo")):
            user_id = i + 1
            if user_id not in valid_nodes:
                continue
            for conn in conexiones:
                if user_id != conn and conn <= max_nodos and conn in valid_nodes:
                    G.add_edge(user_id, conn)
        print(f"✅ Grafo construido con {G.number_of_nodes():,} nodos y {G.number_of_edges():,} aristas.\n")
        self.graph = G
        self.num_nodes = G.number_of_nodes()
        self.num_edges = G.number_of_edges()
        logger.info(f"Grafo construido exitosamente:")
        logger.info(f"  - Nodos: {G.number_of_nodes():,}")
        logger.info(f"  - Aristas: {G.number_of_edges():,}")
        logger.info(f"  - Densidad: {nx.density(G):.6f}")
        logger.info(f"  - Memoria: {self._memory_usage()}")
        return G

    def load_adjacency_batch(self, file_obj, start_idx: int, batch_size: int) -> list:
        current_idx = start_idx
        batch_lines = []
        for _ in range(batch_size):
            line = file_obj.readline()
            if not line:
                break
            line = line.strip()
            if line:
                try:
                    connections = [int(x) for x in line.split(',') if x.strip()]
                    batch_lines.append((current_idx, connections))
                except ValueError as e:
                    logger.warning(f"Error en línea {current_idx}: {e}")
                    batch_lines.append((current_idx, []))
            current_idx += 1
        return batch_lines

    def process_adjacency_list(self) -> int:
        total_edges = 0
        logger.info(f"Procesando adyacencias. Memoria inicial: {self._memory_usage()}")
        try:
            with open(self.user_path,'r') as f:
                user_id = 1
                while True:
                    batch_data = self.load_adjacency_batch(f, user_id, self.batch_size)
                    if not batch_data:
                        break
                    total_edges += sum(len(connections) for _, connections in batch_data)
                    user_id += len(batch_data)
                    del batch_data
                    gc.collect()
            self.num_edges = total_edges
            logger.info(f"Total conexiones: {total_edges}")
            return total_edges
        except Exception as e:
            logger.error(f"Error procesando adyacencias: {str(e)}")
            raise

    def build_complete_sample_graph(self, sample_size: int = 1000) -> nx.DiGraph:
        logger.info(f"Construyendo grafo completo de muestra con {sample_size} nodos")
        G = nx.DiGraph()
        for i in range(1, sample_size + 1):
            G.add_node(i)
        try:
            with open(self.user_path, 'r') as f:
                for i in range(1, sample_size + 1):
                    line = f.readline().strip()
                    if not line:
                        break
                    connections = [int(x) for x in line.split(',') if x.strip()]
                    valid_connections = [c for c in connections if 1 <= c <= sample_size]
                    for conn in valid_connections:
                        G.add_edge(i, conn)
                    if i % 100 == 0:
                        logger.info(f"Procesado nodo {i}/{sample_size}")
            self.graph = G
            logger.info(f"Grafo completo construido: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
            return G
        except Exception as e:
            logger.error(f"Error construyendo grafo completo de muestra: {str(e)}")
            raise

    def plot_network_with_connections(self, sample_size: int = 10000) -> None:
        logger.info(f"Visualizando red con conexiones para {sample_size} nodos sobre mapa mundi")
        if self.graph is None:
            logger.error("Grafo no construido. Ejecute construir_grafo() primero")
            return
        try:
            locations_sample = self.locations_df.head(sample_size).to_pandas()
            fig = go.Figure()
            edge_lats = []
            edge_lons = []
            for edge in self.graph.edges():
                user_from = edge[0]
                user_to = edge[1]
                from_location = locations_sample[locations_sample['user_id'] == user_from]
                to_location = locations_sample[locations_sample['user_id'] == user_to]
                if not from_location.empty and not to_location.empty:
                    from_lat = from_location.iloc[0]['latitude']
                    from_lon = from_location.iloc[0]['longitude']
                    to_lat = to_location.iloc[0]['latitude']
                    to_lon = to_location.iloc[0]['longitude']
                    edge_lats.extend([from_lat, to_lat, None])
                    edge_lons.extend([from_lon, to_lon, None])
            fig.add_trace(go.Scattergeo(
                lat=edge_lats,
                lon=edge_lons,
                mode='lines',
                line=dict(width=1, color='rgba(255, 0, 0, 0.3)'),
                hoverinfo='skip',
                name='Conexiones',
                showlegend=True
            ))
            node_lats = []
            node_lons = []
            node_info = []
            node_colors = []
            node_sizes = []
            for node in self.graph.nodes():
                location = locations_sample[locations_sample['user_id'] == node]
                if not location.empty:
                    lat = location.iloc[0]['latitude']
                    lon = location.iloc[0]['longitude']
                    node_lats.append(lat)
                    node_lons.append(lon)
                    degree = self.graph.degree(node) if isinstance(self.graph, nx.Graph) else self.graph.in_degree(node)
                    out_degree = self.graph.degree(node) if isinstance(self.graph, nx.Graph) else self.graph.out_degree(node)
                    if isinstance(self.graph, nx.Graph):
                        node_info.append(f"Usuario: {node}<br>Conexiones: {degree}<br>Lat: {lat:.3f}, Lon: {lon:.3f}")
                        node_colors.append(degree)
                        node_sizes.append(max(8, min(25, degree + 8)))
                    else:
                        node_info.append(f"Usuario: {node}<br>Seguidores: {degree}<br>Siguiendo: {out_degree}<br>Lat: {lat:.3f}, Lon: {lon:.3f}")
                        node_colors.append(out_degree)
                        node_sizes.append(max(8, min(25, degree + 8)))
            fig.add_trace(go.Scattergeo(
                lat=node_lats,
                lon=node_lons,
                mode='markers',
                hoverinfo='text',
                text=node_info,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(
                        title="Conexiones",
                        x=1.02,
                        len=0.5
                    ),
                    opacity=0.8
                ),
                name='Usuarios',
                showlegend=True
            ))
            fig.update_layout(
                title=dict(
                    text=f"Red Social - {sample_size} usuarios con conexiones en Mapa Mundi",
                    font=dict(size=16),
                    x=0.5
                ),
                mapbox=dict(
                    style="open-street-map",
                    zoom=1,
                    center=dict(lat=0, lon=0)
                ),
                margin=dict(r=0, t=50, l=0, b=0),
                height=700,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            filename = f"network_map_visualization_{sample_size}.html"
            fig.write_html(filename)
            logger.info(f"Visualización de red en mapa guardada como {filename}")
            logger.info(f"Estadísticas del grafo:")
            logger.info(f"- Nodos: {self.graph.number_of_nodes()}")
            logger.info(f"- Aristas: {self.graph.number_of_edges()}")
            logger.info(f"- Densidad: {nx.density(self.graph):.4f}")
            logger.info(f"- Conexiones mostradas en mapa: {len(edge_lats)//3}")
        except Exception as e:
            logger.error(f"Error generando visualización de red: {e}")
            raise

    def plot_mapbox(self, sample_size: int = 1000) -> None:
        logger.info("Visualizando ubicaciones en mapa mundi con Plotly")
        try:
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
            logger.info("Mapa guardado como map_visualization.html")
        except Exception as e:
            logger.error(f"Error generando mapa: {e}")
            raise

class GraphAnalyzer:
    def __init__(self, graph: nx.Graph, locations_df: pl.DataFrame = None):
        self.graph = graph
        self.locations_df = locations_df
        self.graph_dict = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        if locations_df is not None:
            self.locations = {row['user_id']: (row['latitude'], row['longitude']) 
                             for row in locations_df.to_dicts()}
        else:
            self.locations = {}

    def detect_communities_louvain(self) -> dict:
        if isinstance(self.graph, nx.DiGraph):
            G_undirected = self.graph.to_undirected()
        else:
            G_undirected = self.graph
        return community.best_partition(G_undirected)

    def label_propagation(self, max_iter=1000) -> list:
        logger.info("Iniciando detección de comunidades con propagación de etiquetas")
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
        logger.info(f"Se detectaron {len(communities)} comunidades")
        return list(communities.values())

    def bfs_shortest_path(self, start: int, end: int) -> int:
        queue = deque([(start, 0)])
        visited = set([start])
        while queue:
            node, dist = queue.popleft()
            if node == end:
                return dist
            for neighbor in self.graph_dict[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float('inf')

    def average_shortest_path(self, num_samples: int = 10000) -> float:
        logger.info(f"Calculando longitud promedio del camino más corto con {num_samples} muestras")
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

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    class UnionFind:
        def __init__(self, nodes):
            self.parent = {node: node for node in nodes}
            self.rank = {node: 0 for node in nodes}

        def find(self, node):
            if self.parent[node] != node:
                self.parent[node] = self.find(self.parent[node])
            return self.parent[node]

        def union(self, node1, node2):
            root1 = self.find(node1)
            root2 = self.find(node2)
            if root1 != root2:
                if self.rank[root1] > self.rank[root2]:
                    self.parent[root2] = root1
                elif self.rank[root1] < self.rank[root2]:
                    self.parent[root1] = root2
                else:
                    self.parent[root2] = root1
                    self.rank[root1] += 1

    def kruskal_mst(self) -> list:
        if not self.locations:
            logger.error("No se proporcionaron ubicaciones para calcular el MST")
            raise ValueError("Se requiere locations_df para calcular el MST")
        logger.info("Calculando árbol de expansión mínima con Kruskal")
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
        if skipped_edges > 0:
            logger.warning(f"Se omitieron {skipped_edges} aristas debido a nodos sin ubicación")
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

    def analyze_network_properties(self) -> dict:
        logger.info("Analizando propiedades de la red...")
        properties = {}
        properties['num_nodes'] = self.graph.number_of_nodes()
        properties['num_edges'] = self.graph.number_of_edges()
        properties['density'] = nx.density(self.graph)
        if isinstance(self.graph, nx.DiGraph):
            in_degrees = [d for n, d in self.graph.in_degree()]
            out_degrees = [d for n, d in self.graph.out_degree()]
            properties['avg_in_degree'] = np.mean(in_degrees)
            properties['avg_out_degree'] = np.mean(out_degrees)
            properties['max_in_degree'] = max(in_degrees) if in_degrees else 0
            properties['max_out_degree'] = max(out_degrees) if out_degrees else 0
        else:
            degrees = [d for n, d in self.graph.degree()]
            properties['avg_degree'] = np.mean(degrees)
            properties['max_degree'] = max(degrees) if degrees else 0
        if isinstance(self.graph, nx.DiGraph):
            properties['is_strongly_connected'] = nx.is_strongly_connected(self.graph)
            properties['is_weakly_connected'] = nx.is_weakly_connected(self.graph)
            if nx.is_weakly_connected(self.graph):
                properties['num_strongly_connected_components'] = nx.number_strongly_connected_components(self.graph)
        else:
            properties['is_connected'] = nx.is_connected(self.graph)
            properties['num_connected_components'] = nx.number_connected_components(self.graph)
        if isinstance(self.graph, nx.DiGraph):
            pagerank = nx.pagerank(self.graph)
        else:
            pagerank = nx.pagerank(self.graph)
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        properties['top_influential_nodes'] = top_pagerank
        return properties

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

def main():
    start_time = time.time()
    try:
        MAX_NODOS = 1_000
        VIS_NODOS = 1_000
        loader = GraphDataLoader()
        loader.load_locations()
        loader.load_users_data(MAX_NODOS)
        sample_graph = loader.construir_grafo(MAX_NODOS)
        analyzer = GraphAnalyzer(sample_graph, loader.locations_df)
        properties = analyzer.analyze_network_properties()
        logger.info("Propiedades de la red:")
        for key, value in properties.items():
            if key != 'top_influential_nodes':
                logger.info(f"  {key}: {value}")
        logger.info("Top 5 nodos más influyentes (PageRank):")
        for node, score in properties['top_influential_nodes'][:5]:
            logger.info(f"  Usuario {node}: {score:.4f}")
        communities = analyzer.detect_communities_louvain()
        logger.info(f"Comunidades detectadas (Louvain): {len(set(communities.values()))}")
        communities_prop = analyzer.label_propagation()
        logger.info(f"Comunidades detectadas (Propagación de etiquetas): {len(communities_prop)}")
        for i, comm in enumerate(communities_prop[:5], 1):
            logger.info(f"  Comunidad {i}: {len(comm)} nodos (primeros 5: {comm[:5]})")
        avg_path_length = analyzer.average_shortest_path(num_samples=1000)
        logger.info(f"Longitud promedio del camino más corto: {avg_path_length:.2f}")
        try:
            mst = analyzer.kruskal_mst()
            logger.info(f"Árbol de expansión mínima: {len(mst)} aristas")
            for edge in mst[:5]:
                logger.info(f"  Arista: {edge[0]} -> {edge[1]}, peso: {edge[2]:.2f} km")
        except KeyError as e:
            logger.error(f"Error en kruskal_mst: Nodo {e} no tiene ubicación asociada")
        except Exception as e:
            logger.error(f"Error en kruskal_mst: {str(e)}")
        origen, destino = 1, min(500, MAX_NODOS)
        if origen in sample_graph.nodes and destino in sample_graph.nodes:
            dist, path = analyzer.find_shortest_path_dijkstra(origen, destino)
            logger.info(f"Dijkstra {origen} -> {destino}: distancia = {dist}, camino = {path[:10]}...")
        logger.info(f"Generando visualización con {VIS_NODOS} nodos...")
        loader.plot_network_with_connections(VIS_NODOS)
        loader.plot_mapbox(min(1000, MAX_NODOS))
        logger.info(f"Proceso finalizado en {time.time() - start_time:.2f}s")
        logger.info("Archivos generados:")
        logger.info(f"  - network_map_visualization_{VIS_NODOS}.html (Red con conexiones sobre mapa mundi)")
        logger.info("  - map_visualization.html (Mapa de ubicaciones)")
    except Exception as e:
        logger.error(f"Error en main: {str(e)}")
        raise

if __name__ == "__main__":
    main()