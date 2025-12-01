from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

def oc(request):
    return render(request,"opencf.html")

import math

class Openchannalflow:
    
    def __init__(self,calculation_type,basewidth=0,height=0,slope1=0,slope2=0,cslope=0,Manning_coefficient=1,velocity=0,discharge=0):
        self.calculation_type = calculation_type
        self.b = basewidth
        self.y = height
        self.z1 = slope1
        self.z2 = slope2
        self.s = cslope
        self.n = Manning_coefficient
        self.V = velocity
        self.Q = discharge

    def Trapezoid(self):
        t = self.b + self.y*(self.z1 + self.z2)
        A = (self.y*(self.b + t))/2
        P = self.b + self.y * (math.sqrt(1+self.z1**2) + math.sqrt(1+self.z2**2))
        R = A/P
        D=A/t       
        self.different_calculations(R,A) 
        result = {
            'z1': f"{self.z1:.4f}", 'z2': f"{self.z2:.4f}", 'bottom_width': f"{self.b:.4f}", 'water_depth': f"{self.y:.4f}",
            'channel_slope': f"{self.s:.4f}", 'flow_area': f"{A:.4f}", "wetted_perimeter": f"{P:.4f}", "hydraulic_radius": f"{R:.4f}",
            "topwidth": f"{t:.4f}", "hydraulic_depth": f"{D:.4f}", "velocity": f"{self.V:.4f}", "flow": f"{self.Q:.4f}","n": f"{self.n:.4f}"
        }
        return result


    def Rectangle(self):
        A = self.b*self.y
        P = self.b+2*self.y
        R = A/P
        t = self.b
        D=A/t
        self.different_calculations(R,A)
        result = {
            'bottom_width': f"{self.b:.4f}", 'water_depth': f"{self.y:.4f}", 'channel_slope': f"{self.s:.4f}", 'flow_area': f"{A:.4f}",
            "wetted_perimeter": f"{P:.4f}", "hydraulic_radius": f"{R:.4f}", "topwidth": f"{t:.4f}", "hydraulic_depth": f"{D:.4f}",
            "velocity": f"{self.V:.4f}", "flow": f"{self.Q:.4f}","n": f"{self.n:.4f}"
        }
        return result


    def Triangle(self):
        A = 0.5*(self.z1 + self.z2)*(self.y**2)
        P = self.y*(math.sqrt(1+self.z1**2) + math.sqrt(1+self.z2**2))
        R = A/P
        t = self.z1*self.y + self.z2*self.y
        D=A/t
        self.different_calculations(R,A)
        result = {
            'z1': f"{self.z1:.4f}", 'z2': f"{self.z2:.4f}", 'bottom_width': f"{self.b:.4f}", 'water_depth': f"{self.y:.4f}",
            'channel_slope': f"{self.s:.4f}", 'flow_area': f"{A:.4f}", "wetted_perimeter": f"{P:.4f}", "hydraulic_radius": f"{R:.4f}",
            "topwidth": f"{t:.4f}", "hydraulic_depth": f"{D:.4f}", "velocity": f"{self.V:.4f}", "flow": f"{self.Q:.4f}","n": f"{self.n:.4f}"
        }        
        return result
    
    def Circle(self):
        a_c = math.sinh( ( (self.b/2) - self.y)  / (self.b/2) )
        A = (a_c*3.14*(self.b**2) ) /180
        P = a_c*self.b
        R = A/P
        t = math.sin(a_c)*self.b
        D=A/t
        self.different_calculations(R,A)
        result = {
             'bottom_width': f"{self.b:.4f}", 'water_depth': f"{self.y:.4f}",
            'channel_slope': f"{self.s:.4f}", 'flow_area': f"{A:.4f}", "wetted_perimeter": f"{P:.4f}", "hydraulic_radius": f"{R:.4f}",
            "topwidth": f"{t:.4f}", "hydraulic_depth": f"{D:.4f}", "velocity": f"{self.V:.4f}", "flow": f"{self.Q:.4f}","n": f"{self.n:.4f}"
        } 
        return result

    def different_calculations(self,R,A):
        if self.calculation_type == "velocity_discharge":
            self.V = (1/self.n)*(R**(2/3))*(self.s**0.5)
            self.Q = A*self.V
            
        elif self.calculation_type == "channel_slope_v":
            self.s = (self.n**2)*(self.V**2)/(R**(2/3))
            self.Q = A*self.V
        
        elif self.calculation_type == "channel_slope_q":
            self.s = (self.n**2)*(self.Q**2)/((R**(2/3))*(A**2))
            self.V = self.Q/A

        elif self.calculation_type == "manning_v":
            self.n = (self.s**0.5)*(R**(2/3))/self.V
            self.Q = A*self.V
        
        elif self.calculation_type == "manning_q":
            self.n = (self.s**0.5)*(R**(2/3))/(self.Q/A)
            self.V = self.Q/A
        
        # elif self.calculation_type == "depth_q":
        #     pass


def calculate(request):
    if request.method == 'POST':
        calculation_type = request.POST.get('calculation_type')
        channel_type = request.POST.get('channel_type')
        basewidth = float(request.POST.get('bottom_width', 0))
        try:
            height = float(request.POST.get('water_depth', 0))
        except:
            height = 0
        
        try:
            slope1 = float(request.POST.get('z1', 0))
            slope2 = float(request.POST.get('z2', 0))
        except:
            slope1 = 0
            slope2 = 0
        cslope = float(request.POST.get('channel_slope', 0))
        try:
            Manning_coefficient = float(request.POST.get('n', 1))
        except:
            Manning_coefficient = float(request.POST.get('n', 1))
        velocity = request.POST.get('velocity')
        discharge = request.POST.get('flow')
        
        # print(f"Inputs: {calculation_type}, {channel_type}, {basewidth}, {height}, {slope1}, {slope2}, {cslope}, {Manning_coefficient}, {velocity}, {discharge}")
                                                 

        if velocity is not None:
            velocity = float(velocity)
        else:
            velocity = 0

        if discharge is not None:
            discharge = float(discharge)
        else:
            discharge = 0

        if calculation_type == 'velocity_discharge':
            a = Openchannalflow(calculation_type, basewidth, height, slope1, slope2, cslope, Manning_coefficient)
        
        elif calculation_type == 'channel_slope_v':
            a = Openchannalflow(calculation_type, basewidth, height, slope1, slope2, cslope, Manning_coefficient, velocity)
        
        elif calculation_type == 'channel_slope_q':
            a = Openchannalflow(calculation_type, basewidth, height, slope1, slope2, cslope, Manning_coefficient, 0, discharge)

        elif calculation_type == 'manning_v':
            a = Openchannalflow(calculation_type, basewidth, height, slope1, slope2, cslope, 0, velocity)

        elif calculation_type == 'manning_q':
            a = Openchannalflow(calculation_type, basewidth, height, slope1, slope2, cslope, 0,0, discharge)

        elif calculation_type == 'depth_q':
            a = Openchannalflow(calculation_type, basewidth, height, slope1, slope2, cslope, 0, 0, discharge)

        if channel_type == 'Trapezoid':
            result = a.Trapezoid()
        
        elif channel_type == 'Rectangle':
            result = a.Rectangle()

        elif channel_type == 'Triangle':
            result = a.Triangle()

        elif channel_type == 'Circle':
            result = a.Circle()

        
        # print("Result:", result)
        return JsonResponse(result)

    return render(request, 'opencf.html')






