"""Versuch 7: Vielfalt der erzeugten Programme bei identischer Spezifikation."""
from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppgen.generator import generate          # noqa: E402
from ppgen.metrics import analyze             # noqa: E402
from ppgen.spec import Spec                   # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


class Skeleton(ast.NodeTransformer):
    """Abstrahiert Konstanten und Namen: es bleibt die syntaktische Struktur."""
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(0), node)

    def visit_Name(self, node):
        return ast.copy_location(ast.Name("_", node.ctx), node)


def control_skeleton(node, depth=0) -> str:
    out = []
    for s in getattr(node, "body", []) + getattr(node, "orelse", []):
        if isinstance(s, ast.If):
            nb = sum(len(n.values) - 1 for n in ast.walk(s.test) if isinstance(n, ast.BoolOp))
            out.append(f"if{nb}(" + control_skeleton(s, depth + 1) + ")")
        elif isinstance(s, (ast.For, ast.While)):
            out.append(type(s).__name__ + "(" + control_skeleton(s, depth + 1) + ")")
    return ",".join(out)


def main():
    specs = {"klein": Spec(n_params=2, cc=4, max_depth=2, n_loops=1),
             "mittel": Spec(n_params=2, cc=10, max_depth=3, n_loops=2),
             "groß": Spec(n_params=3, cc=20, max_depth=5, n_loops=3)}
    res = {}
    for name, base in specs.items():
        srcs, skel, ctrl, locs = set(), set(), set(), []
        for seed in range(1000):
            d = base.to_dict()
            d["seed"] = seed
            src = generate(Spec.from_dict(d)).source
            srcs.add(src)
            tree = ast.parse(src)
            skel.add(ast.dump(Skeleton().visit(tree)))
            ctrl.add(control_skeleton(tree.body[0]))
            locs.append(analyze(src)["loc"])
        res[name] = {"spec": base.to_dict(), "unique_sources": len(srcs), "unique_skeletons": len(skel),
                     "unique_control": len(ctrl), "loc_mean": float(np.mean(locs)),
                     "loc_min": int(np.min(locs)), "loc_max": int(np.max(locs))}
        print(name, res[name]["unique_sources"], res[name]["unique_skeletons"], res[name]["unique_control"],
              res[name]["loc_mean"], res[name]["loc_min"], res[name]["loc_max"])
    json.dump(res, open(os.path.join(DATA, "diversity.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
