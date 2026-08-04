# Simulacro federado real en Azure — runbook

Monta el experimento federado del TFM como un **escenario de vida real**: 3 máquinas virtuales
(una por silo: Grande / Mediano / Pequeño), cada una con **solo sus propios datos**, entrenando de
forma federada por gRPC contra un coordinador neutral, con el **dashboard en vivo en Azure ML
Studio** (val_loss / WMAPE bajando ronda a ronda) y el modelo global final **registrado en Azure
ML**.

A diferencia de `src/federado_flower.py` (simulación en un solo proceso con Ray), esto usa el
**Deployment Engine** de Flower sobre infraestructura real. El algoritmo es el mismo — se verifica
en el paso 2 antes de gastar en Azure.

## ¿Alcanza el crédito de estudiante ($100)? Sí, con holgura

VMs serie B (CPU; el MLP tiene ~5.000 parámetros, una GPU sería absurda). ~$0,17/hora las 4
encendidas. Una corrida de 15 rondas dura ~15-30 min. Todo el ejercicio (desarrollo + varias
corridas + demo) ≈ **$2-6 de cómputo**. El riesgo no es la tarifa, es **olvidar VMs encendidas** —
por eso hay `auto-shutdown` a las 22:00, `deallocate.sh` (páralo al terminar cada sesión) y
`destroy.sh` (borrado total al final). **Regla de oro: `bash infra/deallocate.sh` al acabar.**

## Prerrequisitos (una vez)

1. **Azure CLI**. Instálalo: <https://learn.microsoft.com/cli/azure/install-azure-cli>
   (Windows: `winget install -e --id Microsoft.AzureCLI`). Luego:
   ```bash
   az login                      # abre el navegador; inicia sesión con la cuenta de estudiante
   az account show               # confirma la suscripción activa
   export AZURE_SUBSCRIPTION_ID="<el id que sale arriba>"
   ```
   (Guarda ese export en un `infra/azure_env.sh` — está gitignored — y haz `source` cada sesión.)
2. **Datos partidos** (local, sin Azure):
   ```bash
   PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe infra/02_split_datos.py
   ```

## Runbook (orden estricto)

| Paso | Comando | Qué hace | ¿Gasta en Azure? |
|---|---|---|---|
| 1 | `python infra/02_split_datos.py` | parte los datos por silo + val_global | no |
| 2 | `cd flower_app && PYTHONPATH=. ../.venv/Scripts/python.exe ../infra/verificar_equivalencia_local.py` | **verifica que la app de despliegue reproduce el resultado documentado** | no |
| 3 | `bash infra/01_provision.sh` | crea RG, red, 4 VMs, workspace, identidad, auto-shutdown | **sí (VMs ON)** |
| 4 | `bash infra/03_deploy.sh` | sube código + datos (cada silo solo el suyo), instala deps | sí |
| 5 | `bash infra/04_start_federation.sh` | arranca SuperLink + 3 SuperNodes | sí |
| 6 | `bash infra/05_run.sh fedavg` | lanza la corrida → **mira Azure ML Studio** | sí |
| 7 | `bash infra/05_run.sh fedprox` | segunda corrida (compara curvas en el dashboard) | sí |
| 8 | (en el coordinador) `python infra/06_registrar_modelo.py` | registra el modelo global en Azure ML | sí |
| 9 | `bash infra/deallocate.sh` | **para el gasto** | baja a ~0 |
| — | `bash infra/destroy.sh` | borrado total (cuando esté todo documentado) | 0 absoluto |

El **paso 2 ya está verificado** en local: la app de despliegue reproduce el val_loss por ronda de
la Condición D documentada (converge a ~0,053-0,055 hacia la ronda 3-5) — misma trayectoria, mismo
algoritmo, solo cambia la infraestructura.

## Ver el dashboard

Azure ML Studio (<https://ml.azure.com>) → tu workspace `ws-tfm-federado` → **Jobs / Experiments**
→ experimento **`tfm-federado-simulacro`**. Cada corrida (fedavg, fedprox) es un run con las
métricas `val_loss` y `wmape_val` por ronda — el gráfico se actualiza en vivo mientras entrena.

## Verificar el aislamiento de datos (el corazón del escenario real)

En cualquier VM de silo, solo existe su propio parquet:
```bash
ssh azureuser@<ip-silo-grande> "ls ~/datos"     # -> grande.parquet   (NO mediano ni pequeno)
```

## Arquitectura (resumen)

```
 vm-coordinador (neutral)  ── SuperLink + ServerApp(FedAvg/FedProx) + evaluate_fn ─► MLflow ─► Azure ML Studio
      ▲   ▲   ▲   (gRPC 9092, IP privada de la VNet)
 vm-silo-grande / -mediano / -pequeno  ── SuperNode + ClientApp, cada uno con SOLO su parquet
```

## Notas

- **Modo inseguro** (sin TLS) porque el tráfico va por la VNet privada de Azure; producción usaría
  TLS (`flower-superlink`/`supernode` con certificados). Documentado como simplificación consciente.
- **Sin secretos:** el coordinador usa *managed identity* para Azure ML; los silos no tocan Azure ML.
- Si `flwr run` (paso 6) da problemas de conexión, revisa `~/superlink.log` en el coordinador y que
  la regla NSG `allow-exec-desde-mi-ip` tenga tu IP pública actual (cambia si cambias de red).
- Ver `docs/RESEARCH_LOG.md` Sesión 30 para el registro completo del ejercicio.
