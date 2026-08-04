"""client_app.py — el ClientApp que corre en cada VM de silo (SuperNode).

Reutiliza `ClienteSilo` de src/federado_flower.py SIN modificarlo: la lógica de `fit()` (épocas
locales, término proximal de FedProx) es idéntica a la de la simulación ya documentada. Lo único
propio del despliegue es de dónde salen los datos y los hiperparámetros: aquí el silo se resuelve
por node-config (cada SuperNode sabe quién es) y su parquet se lee del disco LOCAL de la VM.
"""
from flwr.client import ClientApp
from flwr.common import Context

# Importar task PRIMERO: al cargarse añade ROOT/src al sys.path (honrando TFM_REPO_ROOT), lo que
# permite el import de ClienteSilo de src/ que viene justo después.
from tfm_fl.task import cargar_arquitectura, cargar_datos_silo, resolver_silo
from federado_flower import ClienteSilo  # noqa: E402


def client_fn(context: Context):
    silo = resolver_silo(context.node_config)
    train_df = cargar_datos_silo(silo)
    arq = cargar_arquitectura()

    epocas_locales = int(context.run_config.get("local-epochs", 2))
    lr = float(context.run_config.get("lr", 0.002))

    print(f"[SuperNode {silo}] {len(train_df):,} filas de train locales, "
          f"epocas_locales={epocas_locales}, lr={lr}, arq={arq}")
    return ClienteSilo(silo, train_df, epocas_locales=epocas_locales, lr=lr, arq=arq).to_client()


app = ClientApp(client_fn=client_fn)
