"""
Handler Vercel per FastAPI
Questo file è il punto di ingresso per Vercel
"""

from prediction_main import app

# Vercel cerca questo handler
handler = app