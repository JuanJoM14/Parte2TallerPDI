"""Fase 1 de la Parte 2: partición 70/15/15 del banco de fotos Lego, por bloques.

Qué hace: lee la lista de fotos base de las 4 clases (datos_originales\\ImageSetKey.csv),
las reparte en entrenamiento, validación y prueba, y escribe en particiones\\ los CSV
(train.csv, val.csv, test.csv) y una copia de las fotos por conjunto y por clase.

Regla de partición: por clase, las fotos se agrupan en bloques de 10 minutos tomados del
nombre del archivo (2_<clase>_AAMMDDHHMMSS.jpg). Todas las fotos de un bloque van al mismo
conjunto, porque las fotos seguidas son casi iguales. Los bloques se ordenan
cronológicamente, se mezclan con random.Random(42) y cada uno va al conjunto con mayor
déficit frente a 70/15/15, medido en fotos.

Cómo se ejecuta (desde cualquier carpeta):  python Codigo\\particion.py

Si algo falla antes de crear particiones\\, no se escribe nada. Si particiones\\ ya existe,
el script se detiene sin tocarla. Si falla después de crearla, hay que borrarla completa
antes de volver a correr el script.
"""
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CSV_LISTA = RAIZ / "datos_originales" / "ImageSetKey.csv"
DIR_BASE = RAIZ / "datos_originales" / "Base Images"
DIR_SALIDA = RAIZ / "particiones"

# Nombres de clase tal como están en las carpetas (con guion bajo)
CLASES = ["Brick_1x4", "Brick_2x2_L", "Plate_2x2", "Plate_2x4"]
CONJUNTOS = ["train", "val", "test"]
NOMBRE_CONJUNTO = {"train": "entrenamiento", "val": "validación", "test": "prueba"}
PORCENTAJE = {"train": 70, "val": 15, "test": 15}
TOLERANCIA = 5  # puntos porcentuales por conjunto y clase
SEMILLA = 42
TOTAL_FOTOS = 974
COLUMNAS_CSV = ["archivo", "clase", "bloque"]

# Cifras de la sección 6 de Fase1_Contexto.md:
# clase -> (fotos train, fotos val, fotos prueba, bloques de prueba)
ESPERADO = {
    "Brick_1x4": (169, 40, 29, 2),
    "Brick_2x2_L": (213, 50, 54, 3),
    "Plate_2x2": (153, 31, 23, 1),
    "Plate_2x4": (157, 27, 28, 1),
}

PATRON_NOMBRE = re.compile(r"^2_(?P<clase>.+)_(?P<fecha>\d{12})\.jpg$")


# --- Paso 1 y 2: leer la lista y normalizar nombres de clase ---------------------------

def normalizar_clase(nombre):
    """'Brick 1x4' (CSV) y 'Brick_1x4' (carpeta) pasan a la forma de la carpeta."""
    return nombre.strip().replace(" ", "_")


def leer_lista():
    """Filas 'Base Images' de las 4 clases, con la clase en la forma de la carpeta."""
    fotos = []
    with open(CSV_LISTA, encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            if fila["Folder1"].strip() != "Base Images":
                continue
            clase = normalizar_clase(fila["Brick Type"])
            if clase in CLASES:
                fotos.append({"archivo": fila["Name"], "clase": clase,
                              "carpeta": fila["Folder2"].strip()})
    return fotos


def bloque_de(archivo):
    """AAMMDDHHMM con los minutos redondeados hacia abajo a múltiplo de 10."""
    fecha = PATRON_NOMBRE.match(archivo).group("fecha")  # AAMMDDHHMMSS
    return fecha[:8] + "%02d" % (int(fecha[8:10]) // 10 * 10)


def comprobar_nombres(fotos):
    """El nombre debe seguir el patrón y su clase y carpeta deben coincidir con la columna."""
    problemas = []
    for f in fotos:
        m = PATRON_NOMBRE.match(f["archivo"])
        if not m:
            problemas.append(f"nombre fuera de patrón: {f['archivo']}")
        elif normalizar_clase(m.group("clase")) != f["clase"]:
            problemas.append(f"clase del nombre distinta de la columna: {f['archivo']}")
        elif f["carpeta"] != f["clase"]:
            problemas.append(f"carpeta '{f['carpeta']}' distinta de la clase '{f['clase']}': {f['archivo']}")
    nombres = [f["archivo"] for f in fotos]
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    problemas += [f"nombre repetido en la lista: {n}" for n in repetidos]
    return problemas


def comprobar_total_lista(fotos):
    problemas = []
    if len(fotos) != TOTAL_FOTOS:
        problemas.append(f"fotos en la lista: esperadas {TOTAL_FOTOS}, obtenidas {len(fotos)}")
    for clase in CLASES:
        n = sum(1 for f in fotos if f["clase"] == clase)
        esperado = sum(ESPERADO[clase][:3])
        if n != esperado:
            problemas.append(f"{clase}: esperadas {esperado} fotos, en la lista hay {n}")
    return problemas


def comprobar_faltantes(fotos):
    """Archivos listados que no existen en Base Images\\<clase>\\."""
    return [f"falta en disco: {f['clase']}\\{f['archivo']}" for f in fotos
            if not (DIR_BASE / f["clase"] / f["archivo"]).is_file()]


def comprobar_sobrantes(fotos):
    """Archivos de Base Images\\<clase>\\ que no están en la lista."""
    listados = {(f["clase"], f["archivo"]) for f in fotos}
    problemas = []
    for clase in CLASES:
        for ruta in sorted((DIR_BASE / clase).iterdir()):
            if (clase, ruta.name) not in listados:
                problemas.append(f"sobra en disco (sin listar): {clase}\\{ruta.name}")
    return problemas


# --- Paso 3 y 4: bloques y reparto -------------------------------------------------------

def agrupar_en_bloques(fotos_clase):
    """bloque -> fotos. Las claves ordenadas dan el orden cronológico (AAMMDDHHMM)."""
    bloques = defaultdict(list)
    for f in fotos_clase:
        bloques[bloque_de(f["archivo"])].append(f)
    return bloques


def repartir_clase(fotos_clase):
    """Asigna un conjunto a cada foto de la clase, bloque por bloque."""
    bloques = agrupar_en_bloques(fotos_clase)
    orden = sorted(bloques)  # primero cronológico: la mezcla parte siempre del mismo orden
    # Generador nuevo por clase: el resultado de una clase no depende de las demás.
    random.Random(SEMILLA).shuffle(orden)
    n = len(fotos_clase)
    objetivo = {s: PORCENTAJE[s] / 100 * n for s in CONJUNTOS}  # fotos que debería tener cada conjunto
    acumulado = {s: 0 for s in CONJUNTOS}
    for b in orden:
        # Déficit = objetivo - fotos ya asignadas. max() devuelve el primero de los empatados,
        # y CONJUNTOS está en el orden de desempate: entrenamiento, validación, prueba.
        s = max(CONJUNTOS, key=lambda c: objetivo[c] - acumulado[c])
        acumulado[s] += len(bloques[b])
        for f in bloques[b]:
            f["conjunto"] = s
            f["bloque"] = b


def repartir(fotos):
    for clase in CLASES:
        repartir_clase([f for f in fotos if f["clase"] == clase])


def resumir(fotos):
    """clase -> {'fotos': {conjunto: n}, 'bloques': {conjunto: n}}."""
    resumen = {}
    for clase in CLASES:
        propias = [f for f in fotos if f["clase"] == clase]
        resumen[clase] = {
            "fotos": {s: sum(1 for f in propias if f["conjunto"] == s) for s in CONJUNTOS},
            "bloques": {s: len({f["bloque"] for f in propias if f["conjunto"] == s}) for s in CONJUNTOS},
        }
    return resumen


# --- Paso 5: comprobaciones de la partición ---------------------------------------------

def desviacion(resumen, clase, conjunto):
    """Puntos porcentuales entre lo obtenido y el objetivo (valor absoluto)."""
    n = sum(resumen[clase]["fotos"].values())
    return abs(100 * resumen[clase]["fotos"][conjunto] / n - PORCENTAJE[conjunto])


def comprobar_tolerancia(resumen):
    return [f"{c}, {NOMBRE_CONJUNTO[s]}: desviación {desviacion(resumen, c, s):.1f} > {TOLERANCIA}"
            for c in CLASES for s in CONJUNTOS if desviacion(resumen, c, s) > TOLERANCIA]


def comprobar_bloque_minimo(resumen):
    return [f"{c}: sin bloques en {NOMBRE_CONJUNTO[s]}"
            for c in CLASES for s in CONJUNTOS if resumen[c]["bloques"][s] < 1]


def comprobar_bloques_disjuntos(fotos):
    """Ningún bloque de una clase puede tener fotos en dos conjuntos."""
    conjuntos_de = defaultdict(set)
    for f in fotos:
        conjuntos_de[(f["clase"], f["bloque"])].add(f["conjunto"])
    return [f"{c}, bloque {b}: en {sorted(s)}" for (c, b), s in sorted(conjuntos_de.items()) if len(s) > 1]


def comprobar_total_particion(fotos, resumen):
    problemas = []
    asignadas = sum(sum(resumen[c]["fotos"].values()) for c in CLASES)
    if asignadas != TOTAL_FOTOS or len(fotos) != TOTAL_FOTOS:
        problemas.append(f"total de fotos: esperado {TOTAL_FOTOS}, asignadas {asignadas}, en lista {len(fotos)}")
    sin_conjunto = [f["archivo"] for f in fotos if f.get("conjunto") not in CONJUNTOS]
    problemas += [f"foto sin conjunto: {a}" for a in sin_conjunto]
    return problemas


def comprobar_cifras(resumen):
    """Compara con la tabla de la sección 6 de Fase1_Contexto.md y muestra la diferencia."""
    problemas = []
    for c in CLASES:
        obtenido = (*(resumen[c]["fotos"][s] for s in CONJUNTOS), resumen[c]["bloques"]["test"])
        if obtenido != ESPERADO[c]:
            problemas.append(f"{c}: esperado (train, val, test, bloques test) = {ESPERADO[c]}, obtenido = {obtenido}")
    return problemas


def mostrar_comprobaciones(comprobaciones):
    """Imprime el resultado de cada comprobación y devuelve True si todas pasan."""
    todo_ok = True
    for titulo, problemas in comprobaciones:
        print(f"[{'OK' if not problemas else 'FALLA'}] {titulo}")
        for p in problemas:
            print(f"      {p}")
        todo_ok = todo_ok and not problemas
    return todo_ok


# --- Presentación de cifras ---------------------------------------------------------------

def celda(n, total):
    return f"{n} ({100 * n / total:.0f} %)"


def mostrar_tabla(resumen):
    """Tabla de la sección 6 de Fase1_Contexto.md, con las cifras obtenidas."""
    print("| Clase | Entrenamiento | Validación | Prueba | Bloques de prueba |")
    print("|---|---|---|---|---|")
    total = {s: 0 for s in CONJUNTOS}
    bloques_prueba = 0
    for c in CLASES:
        fotos = resumen[c]["fotos"]
        n = sum(fotos.values())
        print(f"| {c.replace('_', ' ')} | " + " | ".join(celda(fotos[s], n) for s in CONJUNTOS)
              + f" | {resumen[c]['bloques']['test']} |")
        for s in CONJUNTOS:
            total[s] += fotos[s]
        bloques_prueba += resumen[c]["bloques"]["test"]
    n = sum(total.values())
    print("| **Total** | " + " | ".join(f"**{celda(total[s], n)}**" for s in CONJUNTOS)
          + f" | **{bloques_prueba}** |")
    peor = max(((desviacion(resumen, c, s), c, s) for c in CLASES for s in CONJUNTOS))
    print(f"\nDesviación máxima frente al objetivo: {peor[0]:.1f}".replace(".", ",")
          + f" puntos ({peor[1].replace('_', ' ')}, {NOMBRE_CONJUNTO[peor[2]]}), "
          f"dentro de la tolerancia de ±{TOLERANCIA}.")


def mostrar_bloques(resumen):
    """Bloques por conjunto y clase, para compararlos con verif_fase1.py."""
    print("Bloques por clase (train / val / test):")
    for c in CLASES:
        b = resumen[c]["bloques"]
        print(f"  {c.replace('_', ' ')}: {b['train']} / {b['val']} / {b['test']}")


# --- Pasos 6 y 7: escribir CSV y copiar fotos ---------------------------------------------

def fotos_del_conjunto(fotos, conjunto):
    """Orden fijo: clase (como en CLASES) y luego nombre de archivo (= cronológico)."""
    propias = [f for f in fotos if f["conjunto"] == conjunto]
    return sorted(propias, key=lambda f: (CLASES.index(f["clase"]), f["archivo"]))


def guardar_csv(fotos):
    for s in CONJUNTOS:
        with open(DIR_SALIDA / f"{s}.csv", "w", encoding="utf-8", newline="") as f:
            escritor = csv.DictWriter(f, fieldnames=COLUMNAS_CSV, extrasaction="ignore")
            escritor.writeheader()
            escritor.writerows(fotos_del_conjunto(fotos, s))  # archivo = solo el nombre, sin ruta


def copiar_fotos(fotos):
    """Copia cada foto a particiones\\<conjunto>\\<clase>\\ y devuelve cuántas hay por carpeta."""
    cuenta = defaultdict(int)
    for f in fotos:
        destino = DIR_SALIDA / f["conjunto"] / f["clase"]
        destino.mkdir(parents=True, exist_ok=True)
        shutil.copy2(DIR_BASE / f["clase"] / f["archivo"], destino / f["archivo"])
        cuenta[(f["conjunto"], f["clase"])] += 1
    return cuenta


def comprobar_copias(fotos, cuenta):
    """Lo copiado en cada carpeta debe ser igual a lo que dicen los CSV."""
    problemas = []
    for s in CONJUNTOS:
        with open(DIR_SALIDA / f"{s}.csv", encoding="utf-8", newline="") as f:
            filas = list(csv.DictReader(f))
        for c in CLASES:
            en_csv = sum(1 for r in filas if r["clase"] == c)
            en_disco = len([p for p in (DIR_SALIDA / s / c).iterdir() if p.is_file()])
            if not (en_csv == en_disco == cuenta[(s, c)]):
                problemas.append(f"{s}\\{c}: CSV {en_csv}, carpeta {en_disco}, copiadas {cuenta[(s, c)]}")
    return problemas


def mostrar_copias(cuenta):
    print("Archivos copiados por carpeta:")
    for s in CONJUNTOS:
        for c in CLASES:
            print(f"  particiones\\{s}\\{c}\\: {cuenta[(s, c)]}")
        print(f"  particiones\\{s}\\ (total): {sum(cuenta[(s, c)] for c in CLASES)}")


# --- Programa principal ---------------------------------------------------------------------

def main():
    if DIR_SALIDA.exists():
        raise SystemExit(f"particiones\\ ya existe ({DIR_SALIDA}). No se escribió nada. "
                         "La partición se genera una sola vez.")

    fotos = leer_lista()
    print("Paso 1 y 2: lista de fotos y nombres de clase")
    ok = mostrar_comprobaciones([
        (f"Lista con {TOTAL_FOTOS} fotos base de las 4 clases", comprobar_total_lista(fotos)),
        ("Nombres coherentes con la clase y la carpeta", comprobar_nombres(fotos)),
        ("Faltantes = 0 (todo archivo listado existe)", comprobar_faltantes(fotos)),
        ("Sobrantes = 0 (no hay fotos sin listar)", comprobar_sobrantes(fotos)),
    ])
    if not ok:
        raise SystemExit("Comprobaciones del paso 1 fallidas. No se escribió nada.")

    repartir(fotos)
    resumen = resumir(fotos)
    print("\nPaso 5: comprobaciones de la partición")
    ok = mostrar_comprobaciones([
        (f"Proporciones dentro de ±{TOLERANCIA} puntos por clase", comprobar_tolerancia(resumen)),
        ("Al menos 1 bloque por clase en cada conjunto", comprobar_bloque_minimo(resumen)),
        ("Ningún bloque en dos conjuntos", comprobar_bloques_disjuntos(fotos)),
        (f"Total de {TOTAL_FOTOS} fotos, cada una en un solo conjunto", comprobar_total_particion(fotos, resumen)),
        ("Cifras iguales a la sección 6 de Fase1_Contexto.md", comprobar_cifras(resumen)),
    ])
    if not ok:
        raise SystemExit("Comprobaciones del paso 5 fallidas. No se escribió nada.")

    print()
    mostrar_tabla(resumen)
    print()
    mostrar_bloques(resumen)

    # Desde aquí se escribe en disco. Si algo falla, borrar particiones\ completa y repetir.
    DIR_SALIDA.mkdir()
    guardar_csv(fotos)
    cuenta = copiar_fotos(fotos)
    print("\nPaso 6 y 7: CSV y copias")
    ok = mostrar_comprobaciones([("Copias iguales a los CSV en cada carpeta", comprobar_copias(fotos, cuenta))])
    if not ok:
        raise SystemExit("Las copias no coinciden con los CSV. Borrar particiones\\ completa antes de repetir.")
    mostrar_copias(cuenta)


if __name__ == "__main__":
    main()
