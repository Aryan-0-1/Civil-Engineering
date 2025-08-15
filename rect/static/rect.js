function togglePDF() {
    const pdfSection = document.getElementById('pdfSection');
    if (pdfSection.classList.contains('collapsed-pdf')) {
        pdfSection.classList.remove('collapsed-pdf');
        pdfSection.style.height = 'calc(100vh - 100px)'; // Extend to full page height
    } else {
        pdfSection.classList.add('collapsed-pdf');
        pdfSection.style.height = '50px'; // Minimized height
    }
}

function saveReport() {
    html2canvas(document.querySelector("#qChart").parentNode).then(canvas => {
        const imgData = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.href = imgData;
        link.download = 'QvsH_graph.png';
        link.click();
    });
}

if (document.getElementById('qChart')) {
    const ctx = document.getElementById('qChart').getContext('2d');
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Experimental Data',
                    data: Q_values.map((q, i) => ({ x: H_3_2_values[i], y: q })),
                    borderColor: '#2c3e50',
                    backgroundColor: '#3498db',
                    showLine: false
                },
                {
                    label: 'Best Fit Line',
                    data: regression_line.map((y, i) => ({ x: H_3_2_values[i], y: y })),
                    borderColor: '#e74c3c',
                    fill: false,
                    type: 'line'
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: { display: true, text: 'Q vs H³/² Relationship' }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'H³/² (m^1.5)' }
                },
                y: { title: { display: true, text: 'Flow Rate (Q) [m³/s]' } }
            }
        }
    });
}