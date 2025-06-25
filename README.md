# Graph Social Network Analysis

This project provides a Python-based implementation for loading, analyzing, and visualizing large-scale social network graphs with geographical data. It includes functionality for graph construction, community detection, shortest path calculations, and geospatial visualizations.

## Project Structure

- `graph_loader.py`: Main script containing the `GraphDataLoader` and `GraphAnalyzer` classes, along with the `main()` function to orchestrate the workflow.
- Input files (not included):
  - `10_million_location.txt`: CSV file with latitude and longitude for each user.
  - `10_million_user.txt`: Text file with user connections (adjacency list format).
- Output files:
  - `network_map_visualization_<sample_size>.html`: Interactive map showing network connections.
  - `map_visualization.html`: Interactive map showing user locations.
  - `graph_loader.log`: Log file capturing runtime information.

## Dependencies

The project relies on several Python libraries. Install them using the following command in your terminal:

```bash
pip install polars networkx geopandas plotly psutil tqdm numpy shapely python-louvain
```

### Library Descriptions

- **polars**: High-performance DataFrame library for efficient data loading and manipulation.
- **networkx**: Graph library for creating, manipulating, and analyzing complex networks.
- **geopandas**: Extends pandas for geospatial data handling, used for geographical filtering.
- **plotly**: Interactive plotting library for generating web-based visualizations.
- **psutil**: Monitors system resources, used for memory usage tracking.
- **tqdm**: Progress bar for tracking long-running operations.
- **numpy**: Numerical computing library for array operations.
- **shapely**: Geometric operations for spatial analysis.
- **python-louvain**: Implementation of the Louvain algorithm for community detection.

Additionally, a shapefile (`ne_110m_admin_0_countries.shp`) is required for land-based filtering. Download it from [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/).

## Main Components

### `GraphDataLoader` Class

Handles data loading, graph construction, and visualization.

#### Key Methods

- **`__init__(location_path, user_path, batch_size)`**: Initializes the loader with paths to location and user data files and a batch size for processing.
- **`load_locations()`**: Loads and filters geographical locations to ensure they lie on land.
- **`load_users_data(max_users)`**: Loads user connections into memory.
- **`construir_grafo(max_nodos)`**: Constructs an undirected graph using NetworkX.
- **`build_complete_sample_graph(sample_size)`**: Builds a directed graph for a sample of nodes.
- **`plot_network_with_connections(sample_size)`**: Visualizes the network with connections on a world map using Plotly.
- **`plot_mapbox(sample_size)`**: Plots user locations on a world map.
- **`process_adjacency_list()`**: Counts total edges in the user data file.

### `GraphAnalyzer` Class

Performs network analysis on the constructed graph.

#### Key Methods

- **`__init__(graph, locations_df)`**: Initializes with a graph and optional location data.
- **`detect_communities_louvain()`**: Detects communities using the Louvain algorithm.
- **`label_propagation(max_iter)`**: Detects communities using label propagation.
- **`bfs_shortest_path(start, end)`**: Finds the shortest path using Breadth-First Search.
- **`average_shortest_path(num_samples)`**: Estimates average shortest path length.
- **`haversine(lat1, lon1, lat2, lon2)`**: Calculates geographical distance between two points.
- **`kruskal_mst()`**: Computes the Minimum Spanning Tree using Kruskal’s algorithm with geographical weights.
- **`analyze_network_properties()`**: Computes graph metrics (e.g., density, degree, PageRank).
- **`find_shortest_path_dijkstra(start, end)`**: Finds the shortest path using Dijkstra’s algorithm.

### `main()` Function

Orchestrates the workflow:
1. Loads location and user data.
2. Constructs a sample graph.
3. Analyzes network properties, communities, and shortest paths.
4. Generates visualizations.
5. Logs runtime statistics.

## Setup in Visual Studio Code

1. **Clone the Repository** (or create a new project folder):
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install polars networkx geopandas plotly psutil tqdm numpy shapely python-louvain
   ```

4. **Download the Shapefile**:
   - Download the 1:110m Cultural Vectors (Admin 0 – Countries) from [Natural Earth](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/).
   - Extract and place the `ne_110m_admin_0_countries` folder in the project root.

5. **Prepare Input Files**:
   - Place `10_million_location.txt` and `10_million_user.txt` in the project root.
   - Expected formats:
     - `10_million_location.txt`: CSV with no header, columns: latitude, longitude.
     - `10_million_user.txt`: One line per user, comma-separated integers representing connections.

6. **Configure VS Code**:
   - Open the project folder in VS Code.
   - Select the Python interpreter from the virtual environment (`venv`).
   - Install recommended extensions:
     - Python (by Microsoft)
     - Pylance (for better Python IntelliSense)
     - Jupyter (for interactive debugging, if needed)

7. **Run the Script**:
   - Open `graph_loader.py` in VS Code.
   - Run the script using the "Run Python File" button or:
     ```bash
     python graph_loader.py
     ```

## Usage Notes

- **Memory Management**: The script monitors memory usage with `psutil` and processes data in batches to handle large datasets.
- **Logging**: All operations are logged to `graph_loader.log` and the console.
- **Performance**: Adjust `MAX_NODOS` and `VIS_NODOS` in `main()` to balance performance and memory usage.
- **Visualizations**: Output HTML files are interactive and require a web browser to view.

## Troubleshooting

- **Missing Shapefile**: Ensure `ne_110m_admin_0_countries.shp` is in the correct path.
- **Memory Errors**: Reduce `batch_size`, `MAX_NODOS`, or `VIS_NODOS`.
- **Library Issues**: Verify all dependencies are installed in the active virtual environment.
- **File Format Errors**: Ensure input files match the expected format.

## Example Output

After running, check:
- `network_map_visualization_10000.html`: Interactive map with nodes and connections.
- `map_visualization.html`: Map showing user locations.
- `graph_loader.log`: Detailed logs of the process, including graph statistics and errors.