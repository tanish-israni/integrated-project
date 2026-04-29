from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Hospitality App", version="1.0.1")

# Enable CORS for all origins (safe for internal APIs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class AvailabilityResponse(BaseModel):
    available: bool
    rooms_left: int


class BookingRequest(BaseModel):
    room_type: str
    nights: int


class BookingResponse(BaseModel):
    booking_id: str
    status: str


# API Routes (must be defined BEFORE static file mounting)
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/availability", response_model=AvailabilityResponse)
def check_availability(room_type: str = "standard"):
    # Simple, deterministic response for demo purposes.
    rooms_left = 5 if room_type.lower() == "standard" else 2
    return {"available": rooms_left > 0, "rooms_left": rooms_left}


@app.post("/book", response_model=BookingResponse)
def book_room(payload: BookingRequest):
    if payload.nights <= 0:
        raise HTTPException(status_code=400, detail="nights must be > 0")
    return {
        "booking_id": "BK-" + payload.room_type.upper() + "-001",
        "status": "confirmed",
    }


# Serve static files (frontend) - MUST be last
app_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(app_dir, "static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
