from fastapi import FastAPI, HTTPException
from enum import Enum
from pydantic import BaseModel
import api.state as state

app = FastAPI(
    title="API Status Keadaan Seseorang",
    description="API untuk mengupdate dan mendapatkan status keadaan seseorang di ruangan. Status yang didukung: berjalan, jatuh, berdiri, duduk.",
    version="1.0.0"
)

# Enum untuk memastikan hanya status tertentu yang diterima
class StatusEnum(str, Enum):
    berjalan = "berjalan"
    jatuh = "jatuh"
    berdiri = "berdiri"
    duduk = "duduk"

# Model data untuk validasi request body pada endpoint POST
class StatusUpdate(BaseModel):
    status: StatusEnum

# Endpoint POST untuk mengupdate status
@app.post("/status")
async def update_status(status_update: StatusUpdate):
    # Update variabel status yang didefinisikan di state.py
    state.status = status_update.status
    return {"message": "Status updated", "status": state.status}

# Endpoint GET untuk mengambil status saat ini
@app.get("/status")
async def get_status():
    return {"status": state.status}
