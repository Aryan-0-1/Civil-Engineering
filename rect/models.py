from django.db import models

class RectangularNotchReading(models.Model):
    ho = models.FloatField(verbose_name="Datum Height (m)")
    h = models.FloatField(verbose_name="Water Surface Elevation (m)")
    volume = models.FloatField(verbose_name="Volume Collected (L)")  # Changed to 'volume'
    time = models.FloatField(verbose_name="Time (s)")
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def H(self):
        return self.h - self.ho

    @property
    def Q(self):
        return (self.volume / 1000) / self.time  # L to m³ conversion

    @property
    def H_3_2(self):
        return self.H ** 1.5

    @property
    def theoretical_Q(self):
        b = 0.03  # From manual
        g = 9.81
        theoretical_Cd = 0.62  # From manual
        return (2/3) * theoretical_Cd * b * (2*g)**0.5 * self.H_3_2

    @property
    def experimental_Cd(self):
        return self.Q / ((2/3) * 0.03 * (2*9.81)**0.5 * self.H_3_2)

    @property
    def percent_error(self):
        return ((self.experimental_Cd - 0.62)/0.62) * 100

    class Meta:
        ordering = ['created_at']