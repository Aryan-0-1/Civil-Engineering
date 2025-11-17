from django.shortcuts import render
from django.http import JsonResponse
import math
import json

# Example steel sections database
STEEL_SECTIONS = [
    {"name": "ISMB 200", "area": 29.5, "Ix": 1810, "Zx": 181},
    {"name": "ISMB 250", "area": 42.0, "Ix": 4500, "Zx": 360},
    {"name": "ISMB 300", "area": 60.0, "Ix": 8200, "Zx": 550},
]

def page(request):
    return render(request, "design.html")

# Design logic
def design_beam(load, span, steel_grade):
    # Example: simply calculate required section modulus
    # Bending moment = w*L^2 / 8 for simply supported uniformly distributed load
    M = load * span ** 2 / 8  # kNm
    # Allowable stress based on steel grade
    allowable_stress = {"Fe250": 165, "Fe415": 250, "Fe500": 335}[steel_grade]  # MPa
    required_Z = M * 1e6 / allowable_stress  # mm^3

    # Select the lightest section satisfying required Z
    selected = None
    for section in STEEL_SECTIONS:
        if section["Zx"] >= required_Z:
            selected = section
            break
    if not selected:
        selected = {"name": "No suitable section", "area": "-", "Ix": "-", "Zx": "-"}

    return {
        "M_kNm": round(M, 2),
        "required_Z": round(required_Z, 2),
        "selected_section": selected
    }


def design_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        load = float(data.get("load"))
        span = float(data.get("span"))
        steel_grade = data.get("steel_grade")
        result = design_beam(load, span, steel_grade)
        return JsonResponse(result)
    return render(request, "design.html")
