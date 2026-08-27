#### PATCHARAKORN ####

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..aba_builder import build_aba_framework
from ..config import ASPECT_GOLD_FILES
from ..pyarg_service import DEFAULT_SEMANTICS, FAST_SEMANTICS, SEMANTICS, build_aa_view, compute_extensions

router = APIRouter(prefix="/api")


@router.get("/aspects")
def list_aspects() -> dict:
    return {"aspects": sorted(ASPECT_GOLD_FILES)}


@router.get("/graph/{aspect}")
def get_graph(aspect: str, semantics: str = DEFAULT_SEMANTICS) -> dict:
    semantics = semantics or DEFAULT_SEMANTICS  # e.g. a bookmarked "?semantics=" with no value
    """Returns ONE canonical structure covering both views:
      - aba: every assumption/rule/contrary in the framework (nothing hidden)
      - aa:  the real Dung instantiation via framework.generate_af() (constructed
             arguments + derived defeats — genuine AA-level attack graph)
      - extensions: computed by py_arg directly on the same framework object

    Because `aba` and `aa` are both derived from the exact same ABAF instance that gets
    passed to py_arg for `extensions`, there is no way for the rendered picture and the
    evaluated framework to disagree (the bug this app was built to avoid).
    """
    try:
        result = build_aba_framework(aspect)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    framework = result.framework

    aba_rules = [
        {"id": rule.id, "body": sorted(rule.body), "head": rule.head}
        for rule in framework.rules
    ]

    try:
        extensions = compute_extensions(framework, semantics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))

    return {
        "aspect": aspect,
        "semantics": semantics,
        "available_semantics": list(SEMANTICS),
        "fast_semantics": sorted(FAST_SEMANTICS),
        "aba": {
            "assumptions": sorted(framework.assumptions),
            "language": sorted(framework.language),
            "contraries": framework.contraries,
            "rules": aba_rules,
            "atom_roles": result.atom_roles,
        },
        "aa": build_aa_view(framework),
        "extensions": extensions,
        "pairs": result.pairs,
    }
