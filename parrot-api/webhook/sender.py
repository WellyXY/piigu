from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

WEBHOOK_SECRET = "parrot-webhook-secret"


def _sign_payload(payload: dict, secret: str) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


async def send_webhook(
    callback_url: str,
    payload: dict,
    retries: int = 3,
    secret: Optional[str] = None,
):
    if not callback_url:
        return

    sig = _sign_payload(payload, secret or WEBHOOK_SECRET)
    headers = {
        "Content-Type": "application/json",
        "X-Parrot-Signature": sig,
        "X-Parrot-Timestamp": str(int(time.time())),
    }

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(callback_url, json=payload, headers=headers)
                if resp.status_code < 300:
                    logger.info(f"Webhook sent to {callback_url}: {resp.status_code}")
                    return
                logger.warning(f"Webhook {callback_url} returned {resp.status_code} (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"Webhook {callback_url} failed (attempt {attempt + 1}): {e}")

        if attempt < retries - 1:
            import asyncio
            await asyncio.sleep(2 ** attempt)

    logger.error(f"Webhook {callback_url} failed after {retries} attempts")
