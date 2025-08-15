from django import forms
from .models import RectangularNotchReading

class ReadingForm(forms.ModelForm):
    class Meta:
        model = RectangularNotchReading
        fields = ['ho', 'h', 'volume', 'time']  # Match model field names
        widgets = {
            'ho': forms.NumberInput(attrs={'step': '0.001', 'class': 'form-control'}),
            'h': forms.NumberInput(attrs={'step': '0.001', 'class': 'form-control'}),
            'volume': forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
            'time': forms.NumberInput(attrs={'step': '0.1', 'class': 'form-control'}),
        }

    def clean(self):
        cleaned_data = super().clean()
        ho = cleaned_data.get('ho')
        h = cleaned_data.get('h')
        if h <= ho:
            raise forms.ValidationError("Water surface elevation must be greater than datum height")