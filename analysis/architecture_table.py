"""Compact Sensen-inspired architecture exhibit; native editable Word companion."""
from pathlib import Path
import os,json,hashlib
P=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(P/'cache');os.environ['TEMP']=os.environ['TMP']=str(P/'cache')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import redraw_figures as v
from layerwise_architecture import display_specification,SECTION_ROWS,MERGED_VALUE_ROW
def export_table():
    headers,rows,raw_widths=display_specification()
    source=pd.read_csv(P/'Tables/Table_2.csv').fillna('')
    assert list(source.columns)==headers and source.astype(str).values.tolist()==rows
    widths=np.array(raw_widths,dtype=float)*183/sum(raw_widths)
    heights=[5.5]+[5.7 if i in SECTION_ROWS else 4.6 for i in range(len(rows))]
    table_height=sum(heights);height=table_height+8
    fig=plt.figure(figsize=(183/25.4,height/25.4));ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,183),ylim=(0,height));ax.axis('off')
    ax.text(0,height-1,'Table 4. Network architectures and hyperparameters',fontsize=11,fontfamily=v.FONT,va='top',color='black')
    top=table_height;renderer=fig.canvas.get_renderer();overflow=[]
    for i,(values,row_height) in enumerate(zip([headers]+rows,heights)):
        section=i>0 and i-1 in SECTION_ROWS;left=0.
        for j,(value,width) in enumerate(zip(values,widths)):
            if i-1==MERGED_VALUE_ROW and j>1:continue
            if i-1==MERGED_VALUE_ROW and j==1:width=sum(widths[1:])
            bold=i==0 or section;fontsize=10.7 if i==0 else 10.5
            prop=FontProperties(family=v.FONT,size=fontsize,weight='bold' if bold else 'normal')
            actual=renderer.get_text_width_height_descent(str(value),prop,False)[0]/fig.dpi*25.4
            if actual>width-2:overflow.append({'row':i,'column':j,'text':value,'text_mm':actual,'available_mm':width-2})
            align='left' if j==0 else 'center';x=left+1 if align=='left' else left+width/2
            ax.text(x,top-row_height/2,str(value),va='center',ha=align,fontsize=fontsize,
                fontfamily=v.FONT,fontweight='bold' if bold else 'normal',color='black')
            left+=width
        if i==0:ax.hlines(top-row_height,0,183,color='black',lw=.65)
        top-=row_height
    assert not overflow,overflow
    ax.hlines([0,table_height],0,183,color='black',lw=.85)
    stem=P/'Tables/Model_architecture_TNR'
    for extension in ['pdf','svg','png','tiff']:
        kw={'dpi':600,'bbox_inches':'tight','pad_inches':.02}
        if extension=='tiff':kw['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(str(stem)+'.'+extension,**kw)
    fig.savefig(P/'qa/Model_architecture_TNR.png',dpi=180,bbox_inches='tight',pad_inches=.02)
    plt.close(fig)
    (P/'qa/architecture_table_layout.json').write_text(json.dumps({'status':'PASS','type':'Sensen-inspired compact architecture table',
        'manuscript_table':'Table 4','report_table':'Table 2','body_rows':len(rows),'columns':headers,
        'font':'Times New Roman','body_font_pt':10.5,'dimensions_mm':[183,round(height,2)],
        'numeric_width_columns_centered':True,'network_and_hyperparameters_separated':True,
        'exactly_three_rules':True,'black_on_white':True,'text_width_overflows':overflow,
        'editable_in_manuscript':True,'source_csv_sha256':hashlib.sha256((P/'Tables/Table_2.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print('SENSEN_ARCHITECTURE_TABLE_READY: 4 columns, grouped networks, compact hyperparameters, 10.5 pt; no text-width overflows')
if __name__=='__main__':export_table()
