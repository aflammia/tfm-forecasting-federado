"""tfm_fl — Flower App de DESPLIEGUE (Deployment Engine) para el simulacro federado real en Azure.

A diferencia de src/federado_flower.py (Simulation Engine, run_simulation sobre Ray, un solo
proceso), esta app corre sobre una federación real: 1 SuperLink (coordinador) + 3 SuperNodes
(silos) comunicándose por gRPC. Reutiliza el modelo y la lógica de FedAvg/FedProx de src/ sin
duplicarlos -- lo único que cambia es la INFRAESTRUCTURA de ejecución. Ver docs/RESEARCH_LOG.md
Sesión 30 y el plan del simulacro.
"""
