from django.shortcuts import render, redirect
from .forms import ReadingForm
from .models import RectangularNotchReading
import numpy as np
import json
import os
from django.contrib.staticfiles import finders
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.http import FileResponse
from django.shortcuts import render, redirect
from .utils import generate_results_pdf


# @xframe_options_exempt  # Allow iframe embedding for this specific view
def serve_pdf(request):
    pdf_path = 'rect/static/manual_pages_25-33.pdf'
    return FileResponse(open(pdf_path, 'rb'), content_type='application/pdf')


def download_manual(request):
    file_path = finders.find('manual_pages_25-33.pdf')
    if not file_path:
        raise Http404("Manual PDF not found. Please run the experiment first.")

    return FileResponse(
        open(file_path, 'rb'),
        as_attachment=True,
        filename='manual_pages_25-33.pdf'
    )


def download_results(request):
    if RectangularNotchReading.objects.count() < 7:
        return render(request, 'rect.html', {'pdf_error': 'Complete all 7 readings first!'})

    pdf_path = generate_results_pdf()
    return FileResponse(open(pdf_path, 'rb'), as_attachment=True, filename='Experiment_Results.pdf')


def experiment(request):
    if request.method == 'POST':
        num_readings = int(request.POST.get('num_readings', 7))
        if num_readings < 4:
            return render(request, 'rect.html', {
                'error': "Please fill at least four readings.",
                'num_readings': num_readings
            })
        readings = []
        for i in range(1, num_readings + 1):
            try:
                ho = float(request.POST.get(f'ho_{i}', 0))
                h = float(request.POST.get(f'h_{i}', 0))
                volume = float(request.POST.get(f'volume_{i}', 0))
                time = float(request.POST.get(f'time_{i}', 0))
            except ValueError:
                return render(request, 'rect.html', {
                    'error': f"Invalid input for reading {i}. Please ensure all fields are filled correctly.",
                    'num_readings': num_readings
                })
            if time == 0:
                return render(request, 'rect.html', {
                    'error': f"Time cannot be zero for reading {i}. Please correct it.",
                    'num_readings': num_readings
                })
            Q = (volume / time) / 1000
            H = (h - ho) / 1000
            H_3_2 = (H) ** 1.5
            # H= (h - ho)/1000
            experimental_Cd = Q / ((2 / 3) * 0.03 * ((2 * 9.81) ** 0.5) * H_3_2)
            percent_error = ((experimental_Cd - 0.62) / 0.62) * 100
            readings.append({
                'H': H,
                'Q': Q,
                'H_3_2': H_3_2,
                'experimental_Cd': experimental_Cd,
                'percent_error': percent_error

            })

        # Perform regression analysis
        H_3_2_values = [r['H_3_2'] for r in readings]
        Q_values = [r['Q'] for r in readings]
        coeffs = np.polyfit(H_3_2_values, Q_values, 1)
        regression_line = np.polyval(coeffs, H_3_2_values)
        experimental_cd = coeffs[0] * 3 / (2 * 0.03 * (2 * 9.81) ** 0.5)
        regression_eq = f"Q = {coeffs[0]:.4f} * H^(3/2) + {coeffs[1]:.4f}"
        cd_by_regressionequation = f"={(coeffs[0] * 3) / (2 * 0.03 * ((2 * 9.81) ** 0.5)):.4f}"

        context = {
            'readings': readings,
            'H_3_2_values': json.dumps(H_3_2_values),
            'H_values': json.dumps([r['H'] for r in readings]),  # Preprocessed H values
            'Q_values': json.dumps([r['Q'] for r in readings]),  # Preprocessed Q values

            # 'Q_values': json.dumps(Q_values),
            'regression_line': json.dumps(regression_line.tolist()),
            'experimental_cd': experimental_cd,
            'regression_eq': regression_eq,
            'cd_by_regressionequation': cd_by_regressionequation,
            'complete': True,
            'num_readings': num_readings,
            'range': range(1, num_readings + 1)  # Replace num_readings with the variable holding the number of readings

        }
        return render(request, 'rect.html', context)

    # Default case: render the form
    form = ReadingForm(initial={'num_readings': 7})
    return render(request, 'rect.html', {'form': form})
    # def reset(request):
    RectangularNotchReading.objects.all().delete()
    return redirect('experiment')

    # num_readings = int(request.POST.get('num_readings', 7))  # Default to 7 readings
    if request.method == 'POST':
        readings = []
        for i in range(1, num_readings + 1):
            ho = float(request.POST.get(f'ho_{i}', 0))
            h = float(request.POST.get(f'h_{i}', 0))
            volume = float(request.POST.get(f'volume_{i}', 0))
            time = float(request.POST.get(f'time_{i}', 0))
            Q = volume / time
            H_3_2 = h ** 1.5
            experimental_Cd = (Q / (0.62 * (2 / 3) * (2 * 9.81) ** 0.5 * H_3_2))
            readings.append({
                'H': h,
                'Q': Q,
                'H_3_2': H_3_2,
                'experimental_Cd': experimental_Cd
            })

        # Perform regression analysis
        H_3_2_values = [r['H_3_2'] for r in readings]
        Q_values = [r['Q'] for r in readings]
        coeffs = np.polyfit(H_3_2_values, Q_values, 1)
        regression_line = np.polyval(coeffs, H_3_2_values)
        experimental_cd = coeffs[0] * 3 / (2 * 0.03 * (2 * 9.81) ** 0.5)

        context = {
            'readings': readings,
            'H_3_2_values': json.dumps(H_3_2_values),
            'Q_values': json.dumps(Q_values),
            'regression_line': json.dumps(regression_line.tolist()),
            'experimental_cd': experimental_cd,
            'complete': True,
            'num_readings': num_readings
        }
        return render(request, 'rect.html', context)

    return render(request, 'rect.html', {'num_readings': num_readings})


def reset(request):
    RectangularNotchReading.objects.all().delete()
    return redirect('experiment')
    # RectangularNotchReading.objects.all().delete()
    # return redirect('experiment')