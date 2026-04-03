from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import verify_api_key
from api.models import AVAILABLE_POSITIONS, PositionInfo, PositionsResponse

router = APIRouter()

POSITION_DESCRIPTIONS = {
    "blow_job": "Blow job position",
    "cowgirl": "Cowgirl position",
    "doggy": "Doggy style position",
    "handjob": "Handjob position",
    "lift_clothes": "Lift clothes position",
    "masturbation": "Masturbation position",
    "missionary": "Missionary position",
    "reverse_cowgirl": "Reverse cowgirl position",
}


@router.get("/positions", response_model=PositionsResponse)
async def list_positions(_: str = Depends(verify_api_key)):
    return PositionsResponse(
        positions=[
            PositionInfo(name=p, description=POSITION_DESCRIPTIONS.get(p, ""))
            for p in AVAILABLE_POSITIONS
        ]
    )
