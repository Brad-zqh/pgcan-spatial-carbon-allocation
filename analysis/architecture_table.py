"""Readable architecture table, matching editable manuscript Table 4."""
import os
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties
import redraw_figures as v

def export_table():
    data=pd.read_csv(v.HERE/'Tables/Table_2.csv').fillna('')
    widths=np.array([43,63,44,24],float);widths*=183/widths.sum()
    fig=plt.figure(figsize=(183/25.4,2));renderer=fig.canvas.get_renderer()
    def wrap(text,width_mm,bold=False):
        prop=FontProperties(family=v.FONT,size=10.5 if bold else 10.2,weight='bold' if bold else 'normal')
        maximum=(width_mm-3.8)/25.4*fig.dpi
        lines=[];line=''
        for word in str(text).replace('→',' → ').split():
            candidate=(line+' '+word).strip()
            if line and renderer.get_text_width_height_descent(candidate,prop,False)[0]>maximum:
                lines.append(line);line=word
            else:line=candidate
        lines.append(line)
        return '\n'.join(lines)
    rows=[list(data.columns)]+data.astype(str).values.tolist()
    wrapped=[[wrap(text,w,i==0) for text,w in zip(row,widths)] for i,row in enumerate(rows)]
    heights=[max(text.count('\n')+1 for text in row)*4.1+2.2 for row in wrapped]
    total=sum(heights);fig.set_size_inches(183/25.4,total/25.4)
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,183),ylim=(0,total));ax.axis('off')
    top=total
    for i,(row,height) in enumerate(zip(wrapped,heights)):
        left=0
        for j,(text,width) in enumerate(zip(row,widths)):
            ax.text(left+1.9,top-1.1,text,va='top',ha='left',fontsize=10.5 if i==0 else 10.2,
                fontfamily=v.FONT,fontweight='bold' if i==0 or j==0 else 'normal',color='black',linespacing=1.25)
            left+=width
        if i==0:ax.hlines(top-height,0,183,color='black',lw=.7)
        top-=height
    ax.hlines([0,total],0,183,color='black',lw=.8)
    stem=v.HERE/'Tables/Model_architecture_TNR'
    for extension in ['pdf','svg','png','tiff']:
        kwargs={'dpi':600,'bbox_inches':'tight','pad_inches':.015}
        if extension=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(str(stem)+'.'+extension,**kwargs)
    fig.savefig(v.QA/'Model_architecture_TNR.png',dpi=180,bbox_inches='tight',pad_inches=.015)
    plt.close(fig)
    (v.QA/'architecture_table_layout.json').write_text(json.dumps({'type':'Architecture table; not a figure',
        'manuscript_table':'Table 4','report_table':'Table 2','rows':len(data),'font':'Times New Roman',
        'body_font_pt':10.2,'dimensions_mm':[183,round(total,2)],'editable_in_manuscript':True,
        'source_csv_sha256':hashlib.sha256((v.HERE/'Tables/Table_2.csv').read_bytes()).hexdigest()},indent=2),encoding='utf-8')
    print('ARCHITECTURE_TABLE_READY',len(data),'rows; 10.2 pt body; native editable Word table retained.',flush=True)

if __name__=='__main__':export_table()
