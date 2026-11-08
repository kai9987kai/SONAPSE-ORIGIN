"""FastAPI and WebSocket Scientific Observatory Server."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..mathematical_model import (
    GraphStructure,
    HomeostaticStateVector,
    AffectiveStateVector,
    SystemState,
    StateEvolutionOperator,
    NumericalSolver,
)
from ..structure.genome import TopologyFamily
from ..structure.topology import TopologyGenerator
from ..infrastructure.database import ExperimentDatabase


app = FastAPI(title="SONAPSE-ORIGIN Scientific Observatory", version="0.1.0")

# Static files directory
WEB_DIR = Path(__file__).parent / "web"
WEB_DIR.mkdir(parents=True, exist_ok=True)


class TelemetryBroadcaster:
    """Manages active WebSocket connections and streams live simulation telemetry."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.running = False
        self.current_state: Optional[SystemState] = None
        self.operator = StateEvolutionOperator(solver=NumericalSolver.RUNGE_KUTTA_4, seed=42)
        self.db = ExperimentDatabase()
        self.pending_interventions: Dict[str, Any] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for dead in disconnected:
            self.disconnect(dead)

    def initialize_default_sim(self):
        gen = TopologyGenerator(seed=42)
        genome = gen.generate(TopologyFamily.RICH_CLUB, num_nodes=20)
        W = genome.to_matrix()
        edge_index = np.array(
            [[int(k.split("->")[0]) for k in genome.edge_weights.keys()],
             [int(k.split("->")[1]) for k in genome.edge_weights.keys()]],
            dtype=np.int64
        ) if genome.edge_weights else np.zeros((2, 0), dtype=np.int64)

        G = GraphStructure(
            num_nodes=genome.num_nodes,
            edge_index=edge_index,
            edge_weights=np.array(list(genome.edge_weights.values()), dtype=np.float64),
            node_types=genome.node_types,
        )
        X = np.zeros((genome.num_nodes, 4), dtype=np.float64)

        self.current_state = SystemState(
            step=0,
            time=0.0,
            dt=0.05,
            G=G,
            X=X,
            W=W,
            H=HomeostaticStateVector(),
            A=AffectiveStateVector(),
            M={"buffer_size": 0},
            B={"x": 5.0, "y": 5.0},
            E={"ambient_temp": 0.5, "resources": 1.0},
        )


broadcaster = TelemetryBroadcaster()
broadcaster.initialize_default_sim()


@app.get("/")
async def get_index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>SONAPSE-ORIGIN Scientific Observatory Loading...</h3>")


@app.get("/api/state")
async def get_current_state():
    if broadcaster.current_state is None:
        return JSONResponse({"error": "No simulation initialized"}, status_code=400)
    return JSONResponse(broadcaster.current_state.snapshot())


@app.post("/api/intervention")
async def post_intervention(data: Dict[str, Any]):
    broadcaster.pending_interventions.update(data)
    return JSONResponse({"status": "Intervention queued", "interventions": broadcaster.pending_interventions})


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await broadcaster.connect(websocket)
    try:
        while True:
            # Step the simulation forward if there are active connections
            if broadcaster.current_state is not None:
                interventions = broadcaster.pending_interventions.copy()
                broadcaster.pending_interventions.clear()

                broadcaster.current_state = broadcaster.operator.step(
                    broadcaster.current_state,
                    intervention=interventions if interventions else None,
                )

                # Format packet
                nodes_activity = np.mean(np.abs(broadcaster.current_state.X), axis=-1).tolist()
                packet = {
                    "step": broadcaster.current_state.step,
                    "time": round(broadcaster.current_state.time, 3),
                    "nodes_activity": nodes_activity,
                    "node_types": broadcaster.current_state.G.node_types,
                    "homeostasis": {
                        "energy": broadcaster.current_state.H.energy,
                        "integrity": broadcaster.current_state.H.integrity,
                        "temperature": broadcaster.current_state.H.temperature,
                        "fatigue": broadcaster.current_state.H.fatigue,
                        "prediction_error": broadcaster.current_state.H.prediction_error,
                    },
                    "affect": {
                        "valence": broadcaster.current_state.A.valence,
                        "arousal": broadcaster.current_state.A.arousal,
                        "dread": broadcaster.current_state.A.dread,
                        "calmness": broadcaster.current_state.A.calmness,
                        "fear": broadcaster.current_state.A.fear,
                        "joy": broadcaster.current_state.A.joy,
                        "curiosity": broadcaster.current_state.A.curiosity,
                    },
                    "graph": {
                        "num_nodes": broadcaster.current_state.G.num_nodes,
                        "edges": [
                            {"source": int(src), "target": int(dst), "weight": float(broadcaster.current_state.W[src, dst])}
                            for src in range(broadcaster.current_state.G.num_nodes)
                            for dst in range(broadcaster.current_state.G.num_nodes)
                            if abs(broadcaster.current_state.W[src, dst]) > 0.05
                        ],
                    },
                }
                await websocket.send_json(packet)

            await asyncio.sleep(0.08)  # ~12 FPS live stream
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)
    except Exception:
        broadcaster.disconnect(websocket)
