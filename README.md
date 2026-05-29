# OpenGriPS: Open-Source Grid Prediction System

OpenGriPS (Open-Source Grid Prediction System) is an industrial-grade, local-first, distributed computing platform designed for the predictive modeling and multi-modal risk assessment of electrical transmission and distribution infrastructure. By synthesizing multi-scale geophysical fluid dynamics, localized grid telemetry, non-linear machine learning architectures, and real-time multi-source linguistic sentiment streams, OpenGriPS delivers high-fidelity out-of-sample forecasting of grid behavior, thermal line sags, structural stress events, and localized cascading outages.

The platform is engineered under a strict decoupled clean-architecture paradigm, pairing an ultra-low-overhead **C++23 structural optimization and kernel-bypass distributed simulation core** with an asynchronous **Python ASGI (BlackSheep) control plane and data governance gateway**. For analytical flexibility, OpenGriPS isolates rigid administrative orchestration from dynamic exploratory modeling by automating isolated, multi-language **JupyterLab computation sandboxes** featuring kernel bindings for Python, C++ (via `xeus-cling`), and Julia.

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Core Features](#2-core-features)
3. [Engineering & Implementation Details](#3-engineering--implementation-details)
    - [High-Performance C++23 Compute & Hardware Acceleration](#high-performance-c23-compute--hardware-acceleration)
    - [Kernel-Bypass Distributed Interconnect (RDMA)](#kernel-bypass-distributed-interconnect-rdma)
    - [Asynchronous Data Control Plane & Dynamic REST APIs](#asynchronous-data-control-plane--dynamic-rest-apis)
    - [Local-First Analytics & Pluggable Storage Engine](#local-first-analytics--pluggable-storage-engine)
4. [Project Repository Blueprint](#4-project-repository-blueprint)
5. [Dependency Configurations](#5-dependency-configurations)
6. [Installation & Local Deployment](#6-installation--local-deployment)
7. [Licensing & Administrative Governance](#7-licensing--administrative-governance)

---

## 1. System Architecture Overview

OpenGriPS operates as a distributed, multi-tiered computational cockpit. The architecture cleanly segregates concerns into three distinct execution spaces:

```

┌────────────────────────────────────────────────────────────────────────┐
│                      OPENGRIPS CONTROL PLANE (GUI/API)                 │
│   ┌───────────────┐ ┌────────────────────────────────────────────────┐ │
│   │    Projects   │ │ ▶ Launch Project Workspace (JupyterLab)        │ │
│   │   Data Feeds  │ │ ⚡ Topology: 4 Multi-GPU Compute Nodes Active  │ │
│   │    Explorer   │ ├────────────────────────────────────────────────┤ │
│   │   Settings    │ │ 🤖 LLM Data Insight Engine                     │ │
│   └───────────────┘ │ "Analyze anomalous phase metrics during storm" │ │
│                     │ ↳ [DuckDB Vector SQL Engine] -> Synthesis Text │ │
│                     └────────────────────────────────────────────────┘ │
└────────────────────────────────────┬───────────────────────────────────┘
│ (In-Process / Dynamic REST API)
▼
┌────────────────────────────────────────────────────────────────────────┐
│                  CORE HETEROGENEOUS MATHEMATICAL ENGINE                │
│  ┌───────────────────────────────┐  ┌───────────────────────────────┐  │
│  │      Multi-Core CPU Core      │  │      NVIDIA CUDA Kernel       │  │
│  │   C++23 SIMD Vectorization    │  │   TensorRT ML Inference       │  │
│  │  (AVX-512 Matrix Evaluation)  │  │   (Non-Linear Disruption)     │  │
│  └──────────────┬────────────────┘  └───────────────┬───────────────┘  │
└─────────────────┼───────────────────────────────────┼──────────────────┘
│                                   │
└─────────────────┬─────────────────┘
│
▼
┌────────────────────────────────────────────────────────────────────────┐
│                 KERNEL-BYPASS DISTRIBUTED FABRIC (RDMA)                  │
│[Node Alpha (Windows ND)] <═══ Zero-Copy Wire ═══> [Node Beta (Linux RXE)]│
└────────────────────────────────────────────────────────────────────────┘

```

1. **The Administration and Ingestion Control Plane (`src/server/`):** An asynchronous ASGI engine built via `blacksheep` serving as a local secure web browser interface. It supervises active workspace project lifecycles, runs detached multi-source ingestion pipelines, enforces role-based access security, translates linguistic prompt queries via localized LLM adapters, and dynamically exposes runtime analytics tables as operational REST JSON web endpoints.
2. **The Execution Sandbox (`JupyterLab Engine`):** An automated, isolated computational environment provisioned by the control plane on a per-project basis. It maps local virtual project environments allowing researchers to execute code transparently across multiple language kernels (Python, C++23, Julia) and add domain libraries on-the-fly without destabilizing global system environments.
3. **The Heterogeneous Mathematical Engine (`src/engine/`):** A high-performance, native C++23 shared object pipeline exposing highly optimized numerical routines via `pybind11` bindings. It drives multi-threaded execution loops, explicitly binds worker threads to physical CPU rings, vectorizes dense grids via SIMD intrinsics (AVX2/AVX-512), targets local graphics silicon via NVIDIA CUDA kernels, and orchestrates zero-copy cluster communications through hardware kernel-bypass network interfaces.

---

## 2. Core Features

* **Multi-Modal Predictive Synthesizer:** Unifies continuous deterministic physics calculations (e.g., thermal line deformation, atmospheric boundary conditions) with stochastic machine learning models tracking discrete behavioral and operational variables.
* **Linguistic Sentiment Stream Aggregator:** Ingests live unstructured linguistic feeds (RSS feeds, operational dispatches, and public multi-channel text metadata streams) to compute localized operational tension coefficients and track behavioral anomalies.
* **Local-First, Single-File Dataspaces:** Encapsulates each project into a self-contained directory anchored by an in-process, high-throughput database file (`.duckdb`). Relational metadata, time-series measurements, geospatial vectors, and high-dimensional AI vector embeddings reside inside a unified storage container with zero structural background overhead.
* **Cryptographic Remote Control Plane:** Engineered for secure remote operations using a cryptographic key-based SSH daemon layer (`sshd` with password authentication disabled). Automated reverse SSH tunnel mapping provides secure loopback interaction with the background graphical sandboxes via standard local web browsers.
* **Modular Ingestion Factory:** Pulls isolated data ingestion scripts designed as independent Poetry packages dynamically from remote registries. Scripts operate inside isolated dependency sandboxes, executing asynchronously through BlackSheep subprocess lifecycles to guarantee primary platform stability.

---

## 3. Engineering & Implementation Details

### High-Performance C++23 Compute & Hardware Acceleration
The math subsystem avoids old-school abstract header-dependency chains by completely utilizing **C++23 Native Modules** (`.ixx` specification), guaranteeing strict compiler isolation, minimal macro pollution, and ultra-fast build iteration. 

To optimize computing architectures dynamically across **Windows (MSVC)** and **Linux Debian-based (GCC/Clang)** deployment nodes, the compilation framework isolates OS scheduling routines through compile-time CMake preprocessor definitions:
* **Vectorization:** Dense matrix transformations leverage **Eigen3** vector templates, which map explicit SIMD instructions (**AVX-512 / AVX2**) during the compilation loop based on targeted underlying physical architectures.
* **Thread Scheduling:** To eliminate kernel scheduling jitter when processing high-frequency physical fields, the Linux platform implementation (`rt_scheduler.cpp`) leverages native POSIX threads (`pthread`) to lock strict real-time execution parameters (`SCHED_FIFO` with explicit CPU core affinity pinning). The Windows execution path shifts cleanly to Microsoft’s asynchronous Concurrency Runtime (`win_threads.cpp`).
* **Graphics Acceleration:** Highly non-linear ML disruption models and spatial grid evaluations are isolated inside `.cu` compilation units. The system bypasses standard tensor layers by generating highly quantized, localized **TensorRT** runtime models to maximize throughput across active local NVIDIA GPUs.

### Kernel-Bypass Distributed Interconnect (RDMA)
Distributed node synchronization scales past the typical constraints of the traditional TCP/IP network stack. OpenGriPS maps a unified abstraction layer (`rdma_backend.ixx`) to execute **Remote Direct Memory Access (RDMA)** operations, executing zero-copy memory operations directly into remote worker memory spaces without triggering host CPU context interrupts:
* **Linux Deployment Environment:** Links directly against system user-space verbs (`libibverbs` and `librdmacm` via `rdma-core`). In hardware-constrained development environments, the system establishes a virtual software link using **Soft-RoCE (RXE)** bound to the local loopback (`lo`) or a generic Ethernet adapter, providing full hardware emulation for debugging ring buffers and memory registrations (`ibv_reg_mr`).
* **Windows Deployment Environment:** Links dynamically against the Microsoft **NetworkDirect (MS-ND)** Service Provider Interface. Emulated system loops utilize the NetworkDirect Service Provider Simulator (`NDSim`) driver to validate queue pairs and memory pinning locally.

### Asynchronous Data Control Plane & Dynamic REST APIs
The management gateway leverages `blacksheep` as a non-blocking, asynchronous ASGI web hub. The server instantiates user contexts directly within the active request cycle:
* **Dynamic Endpoint Injection:** When researchers finalize predictive mathematical tables inside their sandboxes, they can flag specific data schemas inside the Projects Control Panel. The server service utilizes BlackSheep's dynamic runtime router (`app.router.add_get`) to instantly inject clean REST endpoints (e.g., `/api/v1/projects/{project_id}/{table_name}`) into the live web server, allowing external downstream applications to query active storage tables instantly without rebooting the control plane.
* **Asynchronous Progress Synchronization:** Ingestion tasks and cluster status checks run inside detached asynchronous workers. State updates are propagated reactively to the HTML user interface over low-latency **WebSocket** loops.

### Local-First Analytics & Pluggable Storage Engine
OpenGriPS standardizes on **DuckDB** as its unified transactional and analytical OLAP engine, entirely replacing the need for complex multi-process database installations:
* **Geospatial Processing:** Standardizes on the DuckDB `spatial` extension, treating geographic coordinates, grid topologies, and high-resolution weather polygons as native `GEOMETRY` objects. Spatial intersection and buffering operations occur inside optimized SQL queries using standardized simple features vectors (`ST_Within`, `ST_Intersects`).
* **Semantic Vector Retrieval (RAG):** Document schemas, text logs, and real-time sentiment arrays are indexed directly beside time-series float streams. The control plane implements a Retrieval-Augmented Generation (RAG) framework via the DuckDB `vss` (Vector Similarity Search) module. Unstructured query prompts are passed down to local **Ollama** instances or web API adapters, converted to high-dimensional embedding vectors, and matched using parallelized cosine-similarity calculations (`array_cosine_distance`) straight inside the core SQL execution cycle.
* **Enterprise Scaling Override:** While the system defaults to local file-first tracking, it implements a pluggable storage pattern. Through the integration of DuckDB Foreign Data Wrappers, large-scale systems can check infrastructure override settings to attach, map, and stream real-time tables directly to external data-center clusters like **PostgreSQL**, **ClickHouse** (for multi-terabyte log layers), or **Milvus** (for distributed corporate semantic vectors).

---

## 4. Project Repository Blueprint

```text
OpenGriPS/
│   .gitignore                 # Excludes heavy binaries, local .duckdb dataspaces, and certificates
│   LICENSE                    # Licensing specifications
│   mkdocs.yml                 # Main configuration blueprint for system documentation
│   NOTICE                     # System copyright acknowledgments
│   poetry.lock                # Deterministic lock file for Python virtual spaces
│   pyproject.toml             # Configuration file defining Poetry package spaces and paths
│   README.md                  # System architectural documentation
│
├───docs
│       index.md               # Main repository documentation entrypoint
│
├───src
│   ├───domain                 # Pure Python configuration models and shared schemas
│   │       __init__.py
│   │
│   ├───engine                 # Native C++23 Hardware Accelerated Modules
│   │   │   CMakeLists.txt     # Modern CMake (3.28+) Module Build Configuration
│   │   │   vcpkg.json         # Dependency configuration for native library layers
│   │   │
│   │   ├───bindings
│   │   │       pybind_module.cpp  # Compiles native C++23 modules into importable Python packages
│   │   │       └───protobuf
│   │   │               message_specs.pb.cc
│   │   │               message_specs.pb.h
│   │   │               message_specs.proto   # Binary serialization formats for cluster sync
│   │   │
│   │   ├───include            # Public engine headers
│   │   ├───src
│   │   │   ├───core
│   │   │   │       diff_eq.cpp
│   │   │   │       diff_eq.ixx        # Module Interface: Cross-platform differential solvers
│   │   │   │       matrix_ops.cpp
│   │   │   │       matrix_ops.ixx     # Module Interface: Vectorized SIMD dense math
│   │   │   │
│   │   │   ├───cpu
│   │   │   │       vectorizer.cpp
│   │   │   │       vectorizer.ixx     # Module Interface: Explicit AVX-512 vector handlers
│   │   │   │
│   │   │   ├───cuda
│   │   │   │       grid.cu            # Low-level CUDA code for parallel physical fields
│   │   │   │       inference.cu       # Low-level CUDA code for TensorRT neural operators
│   │   │   │
│   │   │   ├───distributed
│   │   │   │       cluster.ixx        # Module Interface: Cluster state orchestration
│   │   │   │       mpi_worker.cpp     # Multi-node synchronization logic via MPI
│   │   │   │       rdma_backend.cpp
│   │   │   │       rdma_backend.ixx   # Module Interface: Unified zero-copy network bypass
│   │   │   │
│   │   │   └───platform
│   │   │       ├───linux
│   │   │       │       ib_verbs.cpp
│   │   │       │       ib_verbs.ixx   # Module Interface: Linux libibverbs implementations
│   │   │       │       rt_scheduler.cpp # POSIX real-time thread core affinity locks
│   │   │       │
│   │   │       └───win32
│   │   │               nd_verbs.cpp
│   │   │               nd_verbs.ixx   # Module Interface: Windows NetworkDirect implementations
│   │   │               win_threads.cpp # Windows Concurrency Runtime thread scalers
│   │   │
│   │   └───tests
│   │           test_diff_eq.cpp       # Unit validation for numerical core modules
│   │           test_rdma.cpp          # Automated checks for simulated/hardware direct queues
│   │
│   └───server                 # Asynchronous Python Control Plane (BlackSheep Application)
│       │   main.py            # Main application bootloader and server lifecycle middleware
│       │   __init__.py
│       │
│       ├───controllers
│       │       auth.py        # Validates user identities and maps session credentials
│       │       pipeline.py    # Manages asynchronous background subprocess data tasks
│       │       projects.py    # Standard REST endpoints routing project profiles
│       │
│       ├───services
│       │       db.py          # Active project file connector service
│       │       db_manager.py  # DuckDB SQL transaction and spatial execution engines
│       │       ingestion.py   # Discovery service mapping dynamic Poetry data scripts
│       │       jupyter.py     # Asynchronous subprocess management for Jupyter sandboxes
│       │       jupyter_host.py # Secure environment parameterization logic for sandboxes
│       │       llm.py         # Handles prompt token translation and Text-to-SQL logic
│       │       runner.py      # Background pipeline daemon scheduling loops
│       │       vector_store.py # Manages vector similarity generation and RAG pipelines
│       │       __init__.py
│       │
│       ├───static
│       │   └───css
│       │           styles.css # Control Panel structural design system layouts
│       │
│       └───views
│               index.html     # Active Control Cockpit HTML panel view
│               layout.html    # Global Base HTML frame template
└───tests
        __init__.py            # Global core system regression suite tests

```

---

## 5. Dependency Configurations

### Python Core and Sandbox Dependencies (`pyproject.toml`)

```toml
[tool.poetry.dependencies]
python = "^3.11"
blacksheep = "^2.6.2"          # Asynchronous high-performance web framework
uvicorn = { extras = ["standard"], version = "^0.30.0" } # High-speed ASGI server
jinja2 = "^3.1.6"              # Structural UI template engine
duckdb = "^1.5.3"              # Single-file in-process OLAP, spatial, and vector database
perspective-python = "^2.10.0" # WebAssembly-backed real-time streaming data tables
ollama = "^0.2.1"              # Local LLM connectivity interface
pydantic = "^2.7.0"            # Data validation for analytical configuration schemas
pybind11 = "^2.12.0"          # Direct interaction utility for native C++ binaries
protobuf = "^5.26.0"           # High-performance multi-node binary serialization

[tool.poetry.group.dev.dependencies]
jupyterlab = "^4.5.7"          # Core project execution sandbox daemon
ipyleaflet = "^0.20.0"         # Interactive geospatial rendering engine inside notebooks
pydeck = "^0.9.1"              # WebGL-accelerated physical map layers inside notebooks
jupyterlab-nvdashboard = "^0.9.0" # Real-time NVIDIA GPU silicon monitoring panels
voila = "^0.5.7"               # Converts project notebooks to reactive dashboard views
mkdocs = "^1.6.0"              # Global platform documentation system
mkdocs-material = "^9.5.0"     # Material design architecture for system docs
mkdocstrings = { extras = ["python"], version = "^0.25.0" } # Docstring parser

```

### Native Library Layer Configuration (`vcpkg.json`)

```json
{
  "name": "opengrips-core-engine",
  "version-string": "0.1.0",
  "dependencies": [
    "pybind11",
    "protobuf",
    "eigen3",
    "gtest"
  ]
}

```

---

## 6. Installation & Local Deployment

### 1. Pre-requisite Host Configurations

Ensure your target development machine has the native platform drivers and toolchains configured:

* **All Platforms:** Install the [Ollama](https://ollama.com) runtime engine and pull an open-source structural LLM (`ollama pull llama3`).
* **Windows Host Environments:** Install **Visual Studio 2022 (v17.10+)** with C++ Desktop Development components (for C++23 native module parsing) and the official **NVIDIA CUDA Toolkit (v12.x+)**.
* **Linux Debian-Based Environments:** Update to GCC 14 or Clang 18 toolchains and fetch the core communication and distributed clustering libraries:
```bash
sudo apt update && sudo apt install -y build-essential cmake libibverbs-dev librdmacm-dev rdma-core ibverbs-utils openmpi-bin libopenmpi-dev

```



### 2. Python Virtual Environment Bootstrapping

From the root repository directory, initialize your isolated application workspace via Poetry:

```bash
poetry install

```

### 3. Emulating Local Distributed Computing (Linux Soft-RoCE Example)

To safely run and debug zero-copy RDMA network bypass channels locally without enterprise InfiniBand hardware, provision a virtual Soft-RoCE adapter mapped directly onto your system's local loopback network device:

```bash
# Provision virtual adapter rxe0
sudo rdma link add rxe0 type rxe netdev lo

# Confirm system device allocation
ibv_devices

```

### 4. Compiling the Native Acceleration Module

Navigate into the structural engine subfolder to configure and compile the native C++23 `.pyd` or `.so` binary packages via CMake:

```bash
cd src/engine
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=[path_to_vcpkg]/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Release

```

### 5. Launching the OpenGriPS Cockpit

Activate your virtual space and initialize the primary BlackSheep ASGI server:

```bash
poetry shell
python src/server/main.py

```

Open your standard local web browser to `http://localhost:8000` to interact with your local grid management cockpit interface.

---

## 7. Licensing & Administrative Governance

OpenGriPS is distributed under the terms of the Apache License 2.0. All code signatures, module compilation blocks, and documentation modules are subject to standard open-source verification protocols. See the accompanying `LICENSE` and `NOTICE` files for corporate authorship statements, attribution clauses, and dependency redistribution parameters.
