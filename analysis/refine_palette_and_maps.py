"""Rebuild current figures using the requested two-hue colour and exterior coordinates."""
import os
import argparse,json
import redraw_figures as v
import enhanced_analysis as e
import plot_deep_analysis as deep
import plot_benchmark_results as bench
from study_area_map import draw_study_area
from plot_prediction_matrix import draw as density_matrix
from plot_error_diagnostics import draw as error_diagnostics
from plot_allocation_atlas import draw as allocation_atlas

FIGURES={'Figure_1':draw_study_area,'Figure_3':allocation_atlas,'Figure_4':e.enriched_main4,
 'Figure_5':v.fig5,'Figure_6':v.fig6,'Figure_7':deep.figure7,'Figure_8':deep.figure8,
 'Figure_9':deep.figure9,'Figure_10':bench.benchmark_plate,'Figure_11':bench.ablation_plate,
 'Figure_12':bench.graph_plate,'Figure_13':density_matrix,
 'Figure_S1':e.enriched_proxies,'Figure_S2':v.figs2,'Figure_S3':v.figs3,'Figure_S4':e.input_structure,
 'Figure_S5':e.allocation_anatomy,'Figure_S6':e.seed_stability,'Figure_S7':e.benchmark_profiles,'Figure_S8':error_diagnostics}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--figures',nargs='*');parser.add_argument('--exclude',nargs='*',default=[])
    args=parser.parse_args();selected=args.figures or list(FIGURES)
    contract={'backend':'Python','date':'2026-10-21',
        'conclusion':'Keep existing empirical evidence and source values while making the figure family visually consistent and geographically legible.',
        'evidence':'Same saved observations, predictions, metrics and topology; no new training or example values.',
        'archetypes':['Quantitative grids','Hierarchical map/detail/statistics atlas','Native architecture table'],
        'colour_semantics':{'signed':'Pink negative, light neutral at zero, green positive',
            'ordered_nonnegative':'Same pink-neutral-green ramp on the declared positive range; the colour midpoint is not a statistical null',
            'density':'Green sparse outer bins, pink dense core; actual logarithmic cell counts',
            'categorical':'Discrete labelled pink/green shades; no continuous ordering implied'},
        'cartography':'All geographic coordinate labels outside their own frames; edge intersections recomputed for the selected side; physical gutters reserved for small maps.',
        'reference_patterns_adapted':['Map and distribution paired panels','Shared colourbars','Density plots with marginals','Controlled small-multiple comparisons'],
        'reference_handling':'Code and notebooks read as reference material only; supplied setup scripts, models and pipelines are not executed.',
        'export':'600 dpi PNG/TIFF and editable-text PDF/SVG; black Times New Roman text; global title and notes external',
        'verification':'Coordinate label bounding boxes outside each frame; north-arrow containment and protected-feature intersections; current-source/DOCX checks; CVD simulations.'}
    (v.QA/'pink_green_revision_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    old=[a for a in json.loads((v.HERE/'figure_manifest.json').read_text(encoding='utf-8')) if a['figure']!='Figure_14']
    for name in selected:
        if name in args.exclude:continue
        FIGURES[name]()
        changed={a['figure']:a for a in v.manifest}
        current=[changed.pop(a['figure'],a) for a in old]+list(changed.values())
        current.sort(key=lambda entry:(entry['figure'].split('_')[1].startswith('S'),int(entry['figure'].split('_')[1].lstrip('S'))))
        (v.HERE/'figure_manifest.json').write_text(json.dumps(current,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PALETTE_AND_EXTERNAL_COORDINATES_UPDATED',len(v.manifest),flush=True)
