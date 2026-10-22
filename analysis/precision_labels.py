"""User-requested three-decimal display; raw scientific values remain untouched."""
import os
def error_label(value):
    return '<0.001' if 0 < value < .001 else f'{value:.3f}'

def r2_label(value):
    return f'{value:.3f}'

def metric_annotation(name,value):
    return f'{name} < 0.001 pp' if 0 < value < .001 else f'{name} = {value:.3f} pp'
