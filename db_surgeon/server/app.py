"""
FastAPI Server — Serves the DB-Surgeon environment via HTTP/WebSocket.

This is the OpenEnv server that can be:
- Run locally for development
- Containerized via Docker for deployment
- Deployed to HuggingFace Spaces
"""

from __future__ import annotations

import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from db_surgeon.models import DBSurgeonAction, DBSurgeonObservation, DBSurgeonState
from db_surgeon.server.db_surgeon_environment import DBSurgeonEnvironment


def create_app(max_concurrent_envs: int = 64) -> FastAPI:
    """
    Create the FastAPI application for DB-Surgeon.
    
    Args:
        max_concurrent_envs: Maximum number of concurrent environment sessions.
    """
    app = FastAPI(
        title="DB-Surgeon Environment",
        description="OpenEnv RL environment for database schema debugging and repair",
        version="0.1.0",
    )

    # Store active environment sessions
    sessions: dict[str, DBSurgeonEnvironment] = {}

    @app.get("/health")
    async def health():
        return {"status": "ok", "environment": "db-surgeon", "version": "0.1.0"}

    @app.get("/info")
    async def info():
        return {
            "name": "db-surgeon",
            "description": "RL environment for database schema debugging and repair",
            "max_concurrent_envs": max_concurrent_envs,
            "supports_concurrent_sessions": True,
            "action_space": {
                "tools": [
                    "inspect_schema", "run_query", "fix_column",
                    "add_index", "add_constraint", "execute_fix", "submit",
                ],
            },
        }

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await websocket.accept()

        # Create environment for this session
        env = DBSurgeonEnvironment()
        sessions[session_id] = env

        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                method = msg.get("method", "")

                if method == "reset":
                    obs = env.reset()
                    await websocket.send_json({
                        "type": "reset_result",
                        "observation": _obs_to_dict(obs),
                    })

                elif method == "step":
                    action_data = msg.get("action", {})
                    action = DBSurgeonAction(
                        tool_name=action_data.get("tool_name", ""),
                        arguments=action_data.get("arguments", {}),
                    )
                    result = env.step(action)
                    await websocket.send_json({
                        "type": "step_result",
                        "observation": _obs_to_dict(result.observation),
                        "reward": result.reward,
                        "done": result.done,
                        "info": result.info,
                    })

                elif method == "state":
                    state = env.state()
                    await websocket.send_json({
                        "type": "state_result",
                        "state": _state_to_dict(state),
                    })

                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown method: {method}",
                    })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
            except Exception:
                pass
        finally:
            sessions.pop(session_id, None)
            if env._db:
                env._db.reset()

    # HTTP endpoints for simpler interaction (testing/demo)
    _http_env = DBSurgeonEnvironment()

    @app.post("/reset")
    async def http_reset():
        obs = _http_env.reset()
        return {"observation": _obs_to_dict(obs)}

    @app.post("/step")
    async def http_step(action: dict):
        a = DBSurgeonAction(
            tool_name=action.get("tool_name", ""),
            arguments=action.get("arguments", {}),
        )
        result = _http_env.step(a)
        return {
            "observation": _obs_to_dict(result.observation),
            "reward": result.reward,
            "done": result.done,
            "info": result.info,
        }

    @app.get("/state")
    async def http_state():
        state = _http_env.state()
        return {"state": _state_to_dict(state)}

    return app


def _obs_to_dict(obs: DBSurgeonObservation) -> dict:
    return {
        "schema_snapshot": obs.schema_snapshot,
        "error_log": obs.error_log,
        "failing_query": obs.failing_query,
        "last_action_result": obs.last_action_result,
        "step_number": obs.step_number,
        "max_steps": obs.max_steps,
        "action_history": obs.action_history,
    }


def _state_to_dict(state: DBSurgeonState) -> dict:
    return {
        "episode_id": state.episode_id,
        "step_count": state.step_count,
        "initial_bug_type": state.initial_bug_type,
        "root_cause": state.root_cause,
        "is_fixed": state.is_fixed,
        "done": state.done,
        "total_reward": state.total_reward,
    }


# Create default app instance
app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
