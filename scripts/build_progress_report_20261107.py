"""Rebuild the historical ReportLab first draft from saved PN2 measurements.

The current professor report is authored in LaTeX at
docs/reports/progress_20261107/report.tex. This historical builder writes only
to ignored tmp so it cannot overwrite that report.

No model, scorer, reference SWC, reserved data, filtering or experiment is run.
Use the Codex bundled Python with reportlab, Pillow and pypdfium2 installed.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tmp/progress_report_20261107/first_draft_rebuild'
FIG = OUT / 'figures'
TMP = ROOT / 'tmp/progress_report_20261107'
SOURCE_COMMIT = '8c363d3cc3a9c7698d367aa93b9b5d0ce0f63103'
REPO = 'https://github.com/GraphiCortex/PointNeuron_Rec'
PIN = f'{REPO}/blob/{SOURCE_COMMIT}/'
BRANCH = f'{REPO}/blob/pointneuron2/'
NAVY = colors.HexColor('#25445c')
BLUE = colors.HexColor('#3978a6')
TEAL = colors.HexColor('#258176')
ORANGE = colors.HexColor('#b57632')
PURPLE = colors.HexColor('#77579a')
GRAY = colors.HexColor('#65717a')
INK = colors.HexColor('#17212b')
PALE = colors.HexColor('#f1f5f7')
GRID = colors.HexColor('#d9d9d9')

SOURCE_FILES = {
    'S1': ('Cohort and isolation', 'docs/node_cohort_preflight_results.md'),
    'S2': ('Input support audit', 'docs/evidence/curve_transfer_input_support_20261104.json'),
    'S3': ('Transformer four box fit', 'docs/geometric_query_fit_results.md'),
    'S4': ('Curve scoring controls', 'docs/curve_formulation_audit_results.md'),
    'S5': ('Four box transfer', 'docs/curve_transfer_development_results_20261104.md'),
    'S6': ('Broad dense head result', 'docs/broad_curve_head_results_20261105.md'),
    'S7': ('Resampled and compact metrics', 'docs/evidence/resampled_compact_completed_20261105.json'),
    'S8': ('Compact oracle interpretation', 'docs/resampled_compact_results_20261105.md'),
    'S9': ('Stopped selector results', 'docs/stage2_trees_results_20261106.md'),
    'S10': ('Selector metrics and stop receipt', 'docs/evidence/stage2_trees_stopped_20261106.json'),
    'S11': ('Post stop diagnostics', 'docs/post_stage2_diagnostics_results_20261106.md'),
    'S12': ('Post stop portable metrics', 'docs/evidence/post_stage2_diagnostics_completed_20261106.json'),
    'S13': ('Benchmark comparison', 'docs/bigneuron_benchmark_results_20261106.md'),
    'S14': ('Benchmark portable metrics', 'docs/evidence/bigneuron_benchmark_completed_20261106.json'),
    'S15': ('Benchmark per neuron measurements', 'docs/evidence/bigneuron_per_neuron_20261106.csv'),
    'S16': ('Proposal implementation audit', 'docs/proposal_implementation_audit_20261104.md'),
    'S17': ('Editing discussion draft', 'docs/classical_anchored_editing_contract_DRAFT_20261106.md'),
}


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(s):
    return {
        'segment_2': s['segments']['macro']['2.0']['f1'],
        'segment_6': s['segments']['macro']['6.0']['f1'],
        'resampled_2': s['new_metrics']['macro_resampled']['2.0']['f1'],
        'resampled_6': s['new_metrics']['macro_resampled']['6.0']['f1'],
        'length_ratio': s['new_metrics']['pooled_length_ratio'],
    }


def f1(x):
    return f'{100*x:.2f}'


def lr(x):
    return f'{x:.3f}'


def table_values(row):
    return [f1(row[k]) for k in ['segment_2', 'segment_6', 'resampled_2', 'resampled_6']] + [lr(row['length_ratio'])]


def wrap_text(c, text, x, y, width, font='Helvetica', size=10, leading=13, align='left'):
    style = ParagraphStyle('figure', fontName=font, fontSize=size, leading=leading,
                           textColor=INK, alignment=TA_CENTER if align == 'center' else TA_LEFT)
    p = Paragraph(text, style)
    _, height = p.wrap(width, 1000)
    p.drawOn(c, x, y-height)
    return height


def arrow(c, x1, y1, x2, y2, color=GRAY, dashed=False):
    import math
    c.saveState()
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.1)
    if dashed:
        c.setDash(3, 3)
    c.line(x1, y1, x2, y2)
    c.setDash()
    a = math.atan2(y2-y1, x2-x1)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2-6*math.cos(a-.45), y2-6*math.sin(a-.45))
    p.lineTo(x2-6*math.cos(a+.45), y2-6*math.sin(a+.45))
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.restoreState()


def box(c, x, y, w, h, text, color=BLUE, dashed=False, size=10):
    c.saveState()
    c.setFillColor(colors.Color(color.red, color.green, color.blue, alpha=.09))
    c.setStrokeColor(color)
    c.setLineWidth(1)
    if dashed:
        c.setDash(4, 3)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    c.restoreState()
    style = ParagraphStyle('box', fontName='Helvetica', fontSize=size, leading=size+3,
                           alignment=TA_CENTER, textColor=INK)
    p = Paragraph(text, style)
    _, ph = p.wrap(w-16, h)
    assert ph <= h-4, (text, ph, h)
    p.drawOn(c, x+8, y+(h-ph)/2)


def figure(name, width, height, draw):
    path = TMP / (name+'.pdf')
    c = canvas.Canvas(str(path), pagesize=(width, height))
    c.setFillColor(colors.white)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    draw(c)
    c.save()
    doc = pdfium.PdfDocument(str(path))
    image = doc[0].render(scale=3).to_pil()
    image.save(FIG/(name+'.png'))
    doc.close()


def architectures(c):
    wrap_text(c, 'Intended architecture<br/>Research target', 12, 476, 220, 'Helvetica-Bold', 11, 14, 'center')
    wrap_text(c, 'Experimental work<br/>Successive bounded controls', 268, 476, 220, 'Helvetica-Bold', 11, 14, 'center')
    left = [
        ('3D microscopy image', PURPLE, False),
        ('Phase 1<br/>Image-derived point preparation', BLUE, False),
        ('Phase 2<br/>PointNeXt encoder', BLUE, False),
        ('Phase 3<br/>Transformer node proposals', PURPLE, False),
        ('Crop consolidation<br/>and candidate graph', ORANGE, True),
        ('Phase 4<br/>GATv2 connectivity', ORANGE, True),
        ('Graph decoding and whole-image SWC', ORANGE, True),
    ]
    right = [
        ('Same microscopy images<br/>and cached preparation', BLUE, False),
        ('PointNeXt + dense support/offset head<br/>Transformer bypassed in this arm', PURPLE, False),
        ('Saved per-box votes<br/>Consolidation + image-cost forest', BLUE, False),
        ('Classical + Sato + saved learned nodes<br/>Source-labelled candidate pool', BLUE, False),
        ('CPU tree selectors C0 / C1<br/>Stopped by fixed length rule', ORANGE, False),
        ('GATv2 and PointNeXt edge comparison<br/>Not reached', ORANGE, True),
        ('Whole-image merging and final topology<br/>Not built or qualified', ORANGE, True),
    ]
    for x, nodes in [(12, left), (268, right)]:
        for i, (text, color, dashed) in enumerate(nodes):
            y = 391-i*59
            box(c, x, y, 220, 47, text, color, dashed, 9.4)
            if i < len(nodes)-1:
                # The second experimental section reuses saved outputs, not a single trained cascade.
                arrow(c, x+110, y, x+110, y-12, dashed=(x==268 and i==2))
    wrap_text(c, 'Solid boxes: specified design or executed component. Dashed orange boxes: unbuilt or unqualified stages. '
                 'The right column summarizes successive experiments; it is not an adopted end-to-end model.',
              12, 21, 476, size=8.5, leading=11)


def metric_bars(c, aggregate):
    names = ['Classical', 'Archive leader*', 'Broad dense', 'Sato forest']
    keys = ['classical', 'archive', 'broad_dense', 'sato']
    cols = [BLUE, TEAL, PURPLE, ORANGE]
    for pi, (metric, title) in enumerate([('segment_2','Segment F1 at 2 voxels'), ('resampled_2','Resampled F1 at 2 voxels')]):
        x0 = 105+pi*245
        wrap_text(c, title, x0-90, 218, 229, 'Helvetica-Bold', 11, 14, 'center')
        for tick in [0,20,40,60,80,100]:
            x=x0+tick*1.3
            c.setStrokeColor(GRID)
            c.line(x,30,x,185)
            c.setFillColor(GRAY)
            c.setFont('Helvetica',7.5)
            c.drawCentredString(x,19,str(tick))
        for i,(name,key,col) in enumerate(zip(names,keys,cols)):
            y=159-i*36
            c.setFont('Helvetica',8.4)
            c.setFillColor(INK)
            c.drawRightString(x0-7,y+4,name)
            value=100*aggregate[key][metric]
            c.setFillColor(col)
            c.rect(x0,y,value*1.3,15,fill=1,stroke=0)
            c.setFillColor(INK)
            c.drawString(x0+value*1.3+4,y+4,f'{value:.2f}')


def support_chart(c, support):
    d=support['evidence']['partitions']['development']
    wrap_text(c,'Reference length near an input point within 2 voxels',15,199,470,'Helvetica-Bold',12,15,'center')
    for i,(key,label) in enumerate([('all_2','All reference length'),('apparent_thin_dim','Apparent thin and dim')]):
        y=123-i*69
        wrap_text(c,label,10,y+13,145,size=10,leading=13)
        for j,(budget,col) in enumerate([('2048',BLUE),('8192',TEAL)]):
            val=d[budget]['all_2'] if key=='all_2' else d[budget]['strata_2'][key]
            c.setFillColor(col)
            c.rect(166,y-j*22,val*240,15,fill=1,stroke=0)
            c.setFont('Helvetica',9)
            c.setFillColor(INK)
            c.drawString(171+val*240,y-j*22+3,f'{100*val:.2f}%  ({int(budget):,} points)')
    wrap_text(c,'Proximity is an opportunity for reconstruction, not proof of visible signal, correct topology or low user effort.',15,14,470,size=8.5,leading=11)


def oracle_chart(c, oracle):
    labels=['Original union','Compact union','Prune only','Prune plus add']
    keys=['stage0','compact','prune_only','prune_add']
    wrap_text(c,'High segment overlap does not establish compact reconstruction',10,224,480,'Helvetica-Bold',12,15,'center')
    for i,(label,key) in enumerate(zip(labels,keys)):
        x=67+i*122
        s=oracle[key]
        for offset,metric,col in [(-16,'segment_2',BLUE),(13,'resampled_2',TEAL)]:
            v=100*s[metric]
            c.setFillColor(col)
            c.rect(x+offset,60,22,v*1.25,fill=1,stroke=0)
            c.setFillColor(INK);c.setFont('Helvetica',8)
            c.drawCentredString(x+offset+11,65+v*1.25,f'{v:.2f}')
        c.setFont('Helvetica',8.5)
        c.drawCentredString(x+9,42,label)
        c.drawCentredString(x+9,28,f'Length {s["length_ratio"]:.3f}x')
    c.setFillColor(BLUE);c.rect(90,4,8,8,fill=1,stroke=0)
    wrap_text(c,'Segment F1 at 2',102,13,130,size=9,leading=11)
    c.setFillColor(TEAL);c.rect(273,4,8,8,fill=1,stroke=0)
    wrap_text(c,'Resampled F1 at 2',285,13,150,size=9,leading=11)


def review_figure(c, examples):
    # Reflow panels from two saved review sheets using document clipping only.
    # The source PNGs and their image/reference/graph content are unchanged.
    for row,(source,title,description) in enumerate(examples):
        top=550-row*278
        wrap_text(c,title,0,top,500,'Helvetica-Bold',12,15)
        im=PILImage.open(source); w,h=im.size; pw=w/5
        panels=[(0,'Image'),(1,'Reference'),(2,'Retained nodes'),(4,'C1 decoded graph')]
        for col,(idx,label) in enumerate(panels):
            x=col*126; y=top-183; vw=122; vh=151
            wrap_text(c,label,x,top-25,vw,'Helvetica-Bold',9,11,'center')
            # Exclude tiny labels from the saved sheet; repeat them above at readable size.
            source_h=h-28
            scale=min(vw/pw,vh/source_h)
            draw_w=pw*scale; draw_h=source_h*scale
            xx=x+(vw-draw_w)/2; yy=y+(vh-draw_h)/2
            c.saveState()
            p=c.beginPath();p.rect(xx,yy,draw_w,draw_h);c.clipPath(p,stroke=0)
            c.drawImage(str(source),xx-idx*draw_w,yy,w*scale,h*scale)
            c.restoreState()
        wrap_text(c,description,0,top-192,500,size=9.5,leading=13)


def proposed(c):
    box(c,10,395,230,48,'Image + acquisition metadata',PURPLE)
    box(c,10,315,230,56,'Automatic preparation<br/>Image evidence and point cloud',BLUE)
    box(c,275,315,215,56,'Optional sparse review<br/>Include / exclude evidence',TEAL,True)
    arrow(c,240,342,275,342,TEAL,True)
    arrow(c,382,315,382,294,TEAL,True)
    arrow(c,382,294,125,294,TEAL,True)
    arrow(c,125,294,125,315,TEAL,True)
    arrow(c,125,395,125,371)
    box(c,10,224,230,47,'PointNeXt<br/>Image-point features',BLUE)
    box(c,275,224,215,47,'Strong image-derived candidates<br/>Tracer(s) + optional consensus',ORANGE,True)
    arrow(c,125,315,125,271)
    # Accepted image evidence feeds the tracer; corrections are optional.
    c.saveState(); c.setStrokeColor(ORANGE); c.setDash(3,3)
    c.line(240,329,257,329); c.line(257,329,257,247); c.restoreState()
    arrow(c,257,247,275,247,ORANGE,True)
    box(c,10,142,230,53,'Transformer role to discuss<br/>Original proposals OR<br/>candidate-guided refinement',PURPLE,True)
    arrow(c,125,224,125,195)
    c.saveState(); c.setStrokeColor(ORANGE); c.setDash(3,3)
    c.line(275,238,251,238); c.line(251,238,251,169); c.restoreState()
    arrow(c,251,169,240,169,ORANGE,True)
    box(c,275,142,215,53,'Lab-specific preparation profile<br/>Train from accumulated corrections<br/>Validate on separate neurons',TEAL,True,9.2)
    arrow(c,490,343,498,343,TEAL,True)
    arrow(c,498,343,498,168,TEAL,True)
    arrow(c,498,168,490,168,TEAL,True)
    # A validated profile guides preparation of later images, not a live model update.
    c.saveState(); c.setStrokeColor(TEAL); c.setDash(3,3)
    c.line(382,195,382,208); c.line(382,208,266,208)
    c.line(266,208,266,382); c.line(266,382,125,382); c.restoreState()
    arrow(c,125,382,125,371,TEAL,True)
    wrap_text(c,'Validated profile for later images',274,394,213,size=8.2,leading=11)
    box(c,10,64,230,48,'Crop consolidation + candidate graph<br/>GATv2 connectivity',ORANGE,True)
    arrow(c,125,142,125,112)
    box(c,275,64,215,48,'Explicit decoding<br/>Final reconstruction and SWC',ORANGE,True)
    arrow(c,240,88,275,88)
    wrap_text(c,'Dashed: suggested or unbuilt. The lab profile updates preparation for later images; it does not silently update every downstream model. '
                 'Reference SWCs supply training labels and evaluation only, never inference geometry or roots.',10,35,480,size=8.8,leading=12)


styles=getSampleStyleSheet()
styles.add(ParagraphStyle('ReportTitle',fontName='Times-Bold',fontSize=25,leading=29,textColor=colors.black,spaceAfter=12))
styles.add(ParagraphStyle('ReportH1',fontName='Helvetica-Bold',fontSize=16,leading=20,textColor=colors.black,spaceAfter=12))
styles.add(ParagraphStyle('ReportH2',fontName='Helvetica-Bold',fontSize=11.5,leading=15,textColor=colors.black,spaceBefore=9,spaceAfter=5))
styles.add(ParagraphStyle('ReportBody',fontName='Times-Roman',fontSize=11,leading=15,textColor=colors.black,spaceAfter=8))
styles.add(ParagraphStyle('ReportCaption',fontName='Helvetica',fontSize=8.5,leading=11.5,textColor=GRAY,spaceBefore=5,spaceAfter=9))
styles.add(ParagraphStyle('ReportCell',fontName='Helvetica',fontSize=8.6,leading=11,textColor=INK))
styles.add(ParagraphStyle('ReportSmall',fontName='Helvetica',fontSize=8.4,leading=11.2,textColor=INK,spaceAfter=7))


def par(s,style='ReportBody'):
    return Paragraph(s,styles[style])


def tab(headers, rows, widths, font=9, group=0):
    data=[[Paragraph(html.escape(str(v)),ParagraphStyle('th',fontName='Helvetica-Bold',fontSize=font,leading=font+2.5,textColor=colors.white)) for v in headers]]
    for row in rows:
        data.append([Paragraph(str(v),ParagraphStyle('tc',fontName='Helvetica',fontSize=font,leading=font+2.5,textColor=INK)) for v in row])
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    cmd=[('BACKGROUND',(0,0),(-1,0),NAVY),('GRID',(0,0),(-1,-1),.4,GRID),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
         ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),3 if group else 5),('BOTTOMPADDING',(0,0),(-1,-1),3 if group else 5)]
    for i in range(1,len(data)):
        if (i-1)//max(group,1)%2:
            cmd.append(('BACKGROUND',(0,i),(-1,i),PALE))
        if group and (i-1)%group==0:
            cmd.append(('LINEABOVE',(0,i),(-1,i),.9,colors.HexColor('#9aa9b3')))
    t.setStyle(TableStyle(cmd))
    t.spaceAfter=8
    return t


def img(name,width=500):
    p=FIG/(name+'.png')
    im=PILImage.open(p)
    return Image(str(p),width=width,height=width*im.height/im.width)


def main():
    for p in [OUT,FIG,TMP]:p.mkdir(parents=True,exist_ok=True)
    compact=load(SOURCE_FILES['S7'][1])
    bench=load(SOURCE_FILES['S14'][1])
    post=load(SOURCE_FILES['S12'][1])
    stopped=load(SOURCE_FILES['S10'][1])
    support=load(SOURCE_FILES['S2'][1])
    methods={
        'classical':('Classical graph','classical_pruned/development'),
        'broad_dense':('Broad dense head','broad_dense/development'),
        'sato':('Sato top 512 forest','fit4_transfer_sato512/development'),
    }
    aggregate={k:summary(compact['summaries'][v[1]]) for k,v in methods.items()}
    leader=next(v for v in bench['variant_summaries'] if v['variant']=='neutube_updated')
    aggregate['archive']={k:leader[{'segment_2':'segment_f1_2','segment_6':'segment_f1_6','resampled_2':'resampled_f1_2','resampled_6':'resampled_f1_6','length_ratio':'pooled_length_ratio'}[k]] for k in aggregate['classical']}
    cases=[n['case_id'] for n in compact['summaries']['classical_pruned/development']['segments']['neurons']]
    labels=['SLP fly','Frog','Human Allen','Series017','Mouse RGC','Adult zebrafish','Janelia fly']
    aliases=[19,21,35,44,79,94,135]
    long_names=dict(zip(cases,labels))
    detail=[]
    for key,(name,source_key) in methods.items():
        s=compact['summaries'][source_key]
        ns={n['case_id']:n for n in s['new_metrics']['neurons']}
        for n in s['segments']['neurons']:
            cid=n['case_id']; m=ns[cid]
            detail.append({'case_id':cid,'neuron':long_names[cid],'method':key,'segment_2':n['tolerances']['2.0']['f1'],'segment_6':n['tolerances']['6.0']['f1'],'resampled_2':m['resampled']['2.0']['f1'],'resampled_6':m['resampled']['6.0']['f1'],'length_ratio':m['length_ratio'],'status':'completed'})
    with (ROOT/SOURCE_FILES['S15'][1]).open(encoding='utf-8',newline='') as f:
        for r in csv.DictReader(f):
            if r['variant']=='neutube_updated':
                detail.append({'case_id':r['case_id'],'neuron':long_names[r['case_id']],'method':'archive','segment_2':float(r['segment_f1_2']),'segment_6':float(r['segment_f1_6']),'resampled_2':float(r['resampled_f1_2']),'resampled_6':float(r['resampled_f1_6']),'length_ratio':float(r['length_ratio']),'status':r['status']})
    assert len(detail)==28
    assert len({(r['case_id'],r['method']) for r in detail})==28
    # Check all seven-case F1 means against the stored summaries. Length uses pooled lengths.
    for method in aggregate:
        for metric in ['segment_2','segment_6','resampled_2','resampled_6']:
            mean=sum(r[metric] for r in detail if r['method']==method)/7
            assert abs(mean-aggregate[method][metric])<1e-14,(method,metric)
    order=['classical','archive','broad_dense','sato']
    pretty={'classical':'Classical','archive':'Archive leader*','broad_dense':'Broad dense','sato':'Sato forest'}
    detail.sort(key=lambda r:(cases.index(r['case_id']),order.index(r['method'])))
    with (OUT/'development_results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    with (OUT/'development_summary.csv').open('w',encoding='utf-8',newline='') as f:
        rows=[{'method':key,**aggregate[key]} for key in order]
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    oracle={'stage0':summary(compact['summaries']['stage0_union/development']),
            'compact':summary(compact['summaries']['compact_union/development']),
            **{k:summary(v) for k,v in post['editing_summaries'].items()}}
    examples=[
        (ROOT/'tmp/post_stage2_diagnostics_20261106/ridge_review/alias020/XY.png','A  WED SLP training neuron',
         'Diffuse image structure receives many retained nodes and decoded paths beyond the supplied trace. C0/C1 length ratios were 20.19x / 36.59x. This is trace disagreement, not proof that every extra process is biological background.'),
        (ROOT/'tmp/post_stage2_diagnostics_20261106/ridge_review/alias093/XY.png','B  Adult zebrafish training neuron',
         'The reference follows dim branches that remain sparsely represented in retained nodes and C1 output. This view does not isolate sampling, features and selection as causes. Maximum projections cannot establish depth-wise connectivity.'),
    ]
    for source,_,_ in examples:
        dest=FIG/(source.parent.name+'_XY_original.png')
        shutil.copyfile(source,dest)
    figure('architecture_comparison',500,490,architectures)
    figure('input_support',500,210,lambda c:support_chart(c,support))
    figure('development_comparison',500,235,lambda c:metric_bars(c,aggregate))
    figure('oracle_comparison',500,240,lambda c:oracle_chart(c,oracle))
    figure('saved_review_examples',500,562,lambda c:review_figure(c,examples))
    figure('proposed_direction',500,453,proposed)

    pages=[]
    def page(title,items):
        pages.append([par(title,'ReportTitle' if not pages else 'ReportH1')]+items)
    page('PointNeuron 2 research progress report',[
        par('Prepared for Dr Arij<br/>7 November 2026','ReportCaption'),
        par('<b>Purpose.</b> We are investigating whether point-based image features and learned geometry/connectivity can improve neuronal reconstruction from 3D microscopy. This report summarizes the fresh PN2 experiments and requests agreement on the next research direction.'),
        par('<b>Current conclusion.</b> The preparation, training and measurement infrastructure works, but no tested learned reconstruction has demonstrated an advantage over the strongest available automatic tracer on unseen neurons. The intended PointNeXt, Transformer and GATv2 architecture remains a research target. Its experimental bypasses are not adopted replacements.'),
        par('What has been established','ReportH2'),
        par('We built image-derived point preparation, a native PointNeXt prototype, Transformer and dense-head experiments, candidate graphs, CPU selector controls, SWC export and matched geometric scorers. Audits progressively separated execution, fitting, transfer and reconstruction quality. The eight reserved neurons remain untouched. [S1-S10]'),
        tab(['Population','Neurons','Reference-present boxes','Role'],[
            ['Training','33','460','Model fitting and conditional selector CV'],
            ['Development','7','151','Repeatedly inspected diagnostic comparison'],
            ['Reserved','8','Not evaluated','No source images or annotations opened'],
        ],[91,52,123,234],9),
        Spacer(1,9),
        par('The broad dense-head experiment reached <b>36.04% / 67.54%</b> segment F1 at 2 / 6 voxels, compared with <b>54.33% / 69.14%</b> for the matched classical graph. A later archive comparison found a stronger descriptive leader at <b>64.00% / 73.12%</b>, subject to a coordinate-screen limitation. These are per-box results, not an assembled whole-neuron system. [S6, S7, S13]'),
        par('Decision requested','ReportH2'),
        par('We seek agreement on whether to investigate sparse user-guided, lab-adaptive preparation and stronger image-derived candidates; what role the Transformer should play; and how to evaluate the contribution of point features and GATv2. No further training or architecture replacement is implied by this report. Five specific questions appear on page 10.'),
    ])
    page('Architecture and experimental departures',[
        img('architecture_comparison'),
        par('Figure 1. The intended pipeline and the experiments performed. The dense head and later CPU selectors are successive controls, not a validated replacement pipeline. Whole-image assembly and GATv2 were not completed. Reference SWCs supply training labels and evaluation only. [S6, S9, S16]','ReportCaption'),
        par('<b>Transformer status.</b> A geometric-query Transformer fitted the four-box training task successfully. The later dense curve-head arm bypassed its proposal role, with the user\'s agreement for that arm only. This departure from PDF Phase 3 awaits Dr Arij\'s approval. The broad 20-epoch failure was a dense-head result, not a 20-epoch Transformer result. [S3, S6]'),
        par('<b>GATv2 status.</b> GATv2 was retained as the intended connectivity model. Simpler tree selectors were tested first and stopped under their criteria; GATv2 was not trained. No evidence here establishes that GATv2 would succeed or fail. [S9]'),
        par('Other proposal elements remain incomplete: anatomical radius supervision, topology losses and multi-scale Transformer queries. Scaffold SWCs use placeholder radii. These experiments do not implement the complete PDF design. [S16]','ReportCaption'),
    ])
    page('Cohort measurement and verification',[
        par('Why the controls were needed','ReportH2'),
        par('A missing branch can arise from input sampling, proposal localization, selection or graph decoding. We therefore preserved cached inputs and compared saved outputs with no-learning controls before changing architecture. Reference SWCs were used for supervision and measurement, never to choose ordinary inference points, edges or roots. [S1, S4-S6]'),
        tab(['Measure','Definition and practical limit'],[
            ['Segment F1','Directed arclength precision and recall against continuous polylines at 2 and 6 native voxels; maximum quadrature step 0.5 voxel. Close overlapping paths can all score well.'],
            ['Resampled F1','Skeletons uniformly resampled at 2-voxel arclength spacing, then maximum-cardinality one-to-one matching within 2 or 6 voxels. Penalizes extra samples; not a complete topology metric.'],
            ['Length ratio','Predicted arclength / reference arclength. Per-neuron ratios in the detailed table; pooled ratio in aggregate rows. A short graph can still omit branches.'],
            ['Aggregation','Pool box statistics within each neuron, then average the seven neuron F1 values equally. F1 is not recomputed from macro precision and recall.'],
        ],[86,414],9),
        par('Figure 2. Saved input proximity measurements. Increased point count improves two-voxel support, especially in thin/dim strata, but does not prove that downstream learning preserves it. [S2]','ReportCaption'),
        img('input_support',480),
        par('<b>Verification.</b> Saved packets retain source coordinates, intensities and hashes. Synthetic and real-geometry controls checked segment scoring, sampling-phase/subdivision invariance, matching and duplicate sensitivity. Training execution verified optimizer state and exposure counts; these are execution checks, not reconstruction gains. [S3, S4, S6-S8]'),
        par('<b>Scope.</b> The full grids contained 1,964 training and 415 development boxes. The 1,504 / 264 boxes without reference structure are annotation-unknown, not certified background. Tables here score only reference-present boxes. Development identities also appeared in historical PN1 work: they are fresh-PN2 gradient-disjoint diagnostics, not untouched final tests. All coordinates are native voxels; micron views rely on reported calibration and do not establish one physical tolerance across acquisitions. [S1, S11]','ReportCaption'),
    ])
    aggregate_rows=[[pretty[k]]+table_values(aggregate[k]) for k in order]
    page('Development reconstruction comparison',[
        par('Same seven neurons and 151 reference-present boxes. All F1 values are percentages. The learned row is the terminal broad dense head after 9,200 updates, trained on 460 boxes from 33 neurons. [S6, S7, S13-S15]'),
        tab(['System','Seg 2','Seg 6','Res 2','Res 6','Length x'],aggregate_rows,[143,64,64,64,64,101],9.2),
        par('Segment and resampled agreement tell different stories','ReportH2'),
        img('development_comparison'),
        par('Figure 3. The same saved graphs scored at two voxels. Full six-voxel values are in the table. The large drop for dense and Sato forests is consistent with their excess sampled geometry and length, rather than a useful compact trace. [S7, S14]','ReportCaption'),
        par('<b>Classical:</b> automatic whole-guide Otsu thresholding, 3D skeletonization and fixed terminal-spur pruning. <b>Sato:</b> ranked input points, top 512 before the fixed consolidation/forest recipe. <b>Broad dense:</b> PointNeXt support/offset votes, fixed cutoff/consolidation and the same forest family. These controls have different graph sizes; the length column makes that difference explicit. [S4-S7]'),
        par('<b>Archive leader:</b> <i>neutube_updated</i> is the archive label, not the original NeuTube comparator or a reproduced installation. It is the descriptive best of 44 variants on this inspected development set, not a training-selected prospective control. The archive is parameter-optimized, and seed provenance is unresolved. [S13-S15]'),
        par('<b>* Coordinate limitation.</b> The leader\'s Series017 case is zero under the frozen native-bounds screen, not measured zero geometric overlap. Its pooled length is 1.25017x, slightly above 1.25x; rounding is not a gate pass. No retrospective exclusion or rule relaxation was applied. See page 7. [S13]'),
    ])
    rows=[]
    for i,r in enumerate(detail):
        idx=cases.index(r['case_id']); name=f'{labels[idx]}<br/>alias {aliases[idx]}' if i%4==0 else ''
        rows.append([name,pretty[r['method']]]+table_values(r))
    page('All seven development neurons',[
        par('F1 values are percentages at 2 / 6 native voxels. Length is the per-neuron predicted/reference ratio. Each row pools only that neuron\'s scored boxes. [S7, S15]','ReportCaption'),
        tab(['Neuron','System','Seg 2','Seg 6','Res 2','Res 6','Length x'],rows,[92,112,56,56,56,56,72],8.1,4),
        par('* Series017 archive values are protocol zeros caused by the bounds screen, including its reported length contribution. They do not describe the actual supplied SWC\'s overlap or length. No neuron was excluded. Full case IDs, unrounded values and status fields are in development_results.csv.','ReportCaption'),
        par('The classical graph is stronger than the broad dense head on five of seven neurons at two voxels and four of seven at six. The archive leader is contextual evidence with the caveats above; it was not selected before these development comparisons. [S6, S13]','ReportCaption'),
    ])
    cv=stopped['training_CV_summaries']
    cv_rows=[[k]+table_values(summary(cv[k])) for k in ['C0','C1']]
    page('What the learning experiments established',[
        par('Transformer fitting was demonstrated','ReportH2'),
        par('The geometric-query Transformer matched all <b>178 / 178</b> reference target nodes in four training boxes with zero accepted extras at the fixed inference cutoff. The motive was to test whether the model and supervision could fit a small, controlled task. It passed that fitting gate; it did not demonstrate unseen-neuron accuracy, connected reconstruction or whole-image quality. [S3]'),
        par('Dense curve supervision improved recall but failed its broad comparison','ReportH2'),
        par('The dense support/offset experiment replaced exact target-grid supervision with projection to continuous reference curves. Its four-box fit reached 81.49% segment F1 at two voxels, but only 23.20% on development. Broader training raised development F1 to 36.04% and candidate-edge coverage to 81.69%, while two-voxel precision remained 24.80%. [S5, S6]'),
        par('Broad training/development segment F1 at two voxels was 37.84% / 36.04%. The failure therefore cannot be reduced to the earlier small-fit transfer gap. Under the fixed 20-epoch recipe, both precision and useful compactness remained inadequate. This does not prove an asymptotic capacity limit or diagnose the cause as undertraining. [S6]'),
        par('The broad-run gate required segment F1 at least 59.33% / 74.14%, plus wins on at least five of seven neurons at both tolerances. Actual wins were two / three. The run stopped without extension or threshold selection. [S6]'),
        par('Candidate and selector experiments exposed representation problems','ReportH2'),
        par('We then formed classical/Sato/learned candidate pools and tested two edge-label contracts. C0 retained original dense-pool adjacency labels; C1 recomputed adjacency after image-only representative selection. CPU gradient-boosted trees were the simpler control before GATv2. [S9, S10]'),
        tab(['Training CV','Seg 2','Seg 6','Res 2','Res 6','Length x'],cv_rows,[143,64,64,64,64,101],9.2),
        par('Both exceeded the fixed pooled length limit of 1.25x. C1 improved segment overlap while resampled agreement stayed weak and length increased. C0\'s unrounded ratio was 1.254998x, so it failed too. No final selector development comparison, PointNeXt-edge arm or GATv2 training followed. [S9, S10]'),
        par('<b>CV limit.</b> Selector folds left out each training neuron, but upstream learned candidate positions and encoder features had already used all 33 neurons. These are conditional selector-only CV results, not strict full-pipeline leave-neuron-out performance. [S9]','ReportCaption'),
    ])
    page('Oracle headroom and benchmark limitations',[
        img('oracle_comparison'),
        par('Figure 4. All four rows use reference-assisted decisions on the same development boxes. The initial 90.42% segment score passed at 2.40x length. Compaction reduced length and improved one-to-one agreement, correcting its interpretation. These are not learned model scores. [S7, S8, S12]','ReportCaption'),
        tab(['Reference-assisted diagnostic','Seg 2 / 6','Res 2 / 6','Length x'],[
            [label,f'{f1(oracle[k]["segment_2"])} / {f1(oracle[k]["segment_6"])}',f'{f1(oracle[k]["resampled_2"])} / {f1(oracle[k]["resampled_6"])}',lr(oracle[k]['length_ratio'])]
            for k,label in [('stage0','Original union'),('compact','Compact union'),('prune_only','Classical prune only'),('prune_add','Classical prune plus add')]
        ],[170,117,117,96],9),
        par('Prune plus add reached 61.74% resampled F1 at two voxels, clearing the fixed 60% threshold for <b>drafting</b> an editing contract. Prune only reached 58.58%. That gate did not authorize a learner or prove transfer. The add search reached its 128-edit cap in 106 / 151 boxes; 97 used component drops to meet the budget. Its result is an achieved feasible solution, not a solved upper ceiling. [S12, S14, S17]'),
        par('Benchmark coordinate and availability caveats','ReportH2'),
        par('The recovered archive supplied 237 SWCs across 44 variants and seven cases. Of 308 method/case slots, 208 passed, 71 were missing and 29 were rejected. Missing/rejected slots remained zeros; no file was repaired. Archive medians therefore include failure and availability effects. [S13, S14]'),
        par('For the leader\'s Series017 file, one of 16 nodes lies 0.2044 voxel beyond the native voxel-cell bound, failing the fixed 99%-of-vertices screen and zeroing the entire case. That screen is coarse and sampling-density dependent; passing it also does not verify coordinate origin. A new boundary tolerance or offset correction is only a proposal, not an applied measurement fix. [S13]'),
        par('Editing\'s apparent resampled margin over the archive leader is 3.02 points at two voxels and 0.40 at six, and depends on that protocol zero. No strong learned advantage follows. All results remain on record with unchanged gates. [S13, S14]','ReportCaption'),
    ])
    page('Saved image review shows distinct failure modes',[
        img('saved_review_examples'),
        par('Figure 5. Two illustrative training neurons from the saved five-case review, not a random population sample. Four panels are reflowed from each unchanged XY sheet; full five-panel originals are linked in the figure folder. Green: reference; cyan: retained nodes; magenta: C1 decoded graph. C0 appears in the originals. [S11, S12]','ReportCaption'),
        par('Gold-to-ridge diagnostics across all 40 neurons found 16.60% of pooled training reference length and 15.06% of development length more than two voxels from the fixed intensity-ridge proxy. Weak/gapped signal and the proxy itself can cause offsets; this is not proof of wrong gold or permission to relabel. No neuron or annotation was removed. [S11, S12]','ReportCaption'),
    ])
    page('Proposed direction for discussion',[
        img('proposed_direction',480),
        par('Figure 6. A possible research design, not the implemented current pipeline. Sparse corrections and lab adaptation are proposed additions; the original Transformer proposal role versus candidate-guided refinement remains a decision. GATv2 remains intended, but its contribution must be tested. [S16, S17; subsequent design discussion]','ReportCaption'),
        par('A testable claim','ReportH2'),
        par('The original proposal envisaged semi-automatic verification. A testable extension is to learn a lab-specific preparation profile from sparse corrections, improving later uncorrected neurons and reducing correction effort. Improvement only on the corrected image demonstrates interaction, not adaptation.'),
        par('The proposed first test is a training-cohort simulated-user study with bounded include/exclude hints, a fixed simple downstream graph method, zero-correction controls and equal-budget output proofreading. Gold-directed hints are privileged and cannot establish real user time, error-discovery effort or fully automatic accuracy. Stroke extent and width must be bounded, not only their count.'),
        par('Stronger tracer consensus, image-only editing units and a contextual Transformer are options to compare, not automatic additions. Source-group membership is not proof of laboratory independence. New models, QC rules, budgets and gates must be fixed prospectively; reserved neurons remain untouched. No new numerical success criterion is adopted here.'),
    ])
    questions=[
        ('Research objective and interaction',
         'Is a semi-automatic, lab-adaptive front end within the intended PN2 contribution, provided that zero-correction performance, correction effort and transfer to later uncorrected neurons are reported separately?'),
        ('Approval of the architectural departure',
         'Is learned refinement of strong image-derived candidates an acceptable framing? The current experimental arm bypassed Transformer proposals. May that departure be investigated further, while retaining point-feature learning and intended GATv2 connectivity as contributions to test?'),
        ('A meaningful role for the Transformer',
         'Should the next architecture retain independent node proposals, investigate candidate-guided refinement/confidence, or compare the two under one budget? What evidence should justify the Transformer and GATv2 beyond simpler selectors and decoding?'),
        ('Evaluation and reference quality',
         'Can we agree a coordinate/QC protocol and a prospective combination of segment agreement, one-to-one resampled agreement, length and topology checks? Full-cohort results should remain visible, and repeatedly inspected development data should not serve as final confirmation.'),
        ('The next bounded study and final confirmation',
         'Should the next step prioritize simulated sparse guidance and acquisition-group transfer, or stronger tracer-consensus headroom? Which budget and baseline should be fixed before results, and when should a frozen final protocol use the untouched reserved cohort?'),
    ]
    page('Five questions for Dr Arij',[
        par('The requested decisions concern the research framing and next bounded experiment. They do not ask for approval to claim a validated pipeline.'),
        *[item for i,(title,text) in enumerate(questions,1) for item in [par(f'{i}  {title}','ReportH2'),par(text)]],
        par('Status at the report cutoff','ReportH2'),
        par('Stage 2 remains stopped. Its STOP receipt, failed gates, labels, checkpoints and outputs are unchanged. There has been no new GPU run, GATv2 training, neuron exclusion, relabeling, coordinate correction or reserved-data evaluation. The adaptive-front-end and editing designs remain proposals.'),
    ])
    appendix=[
        par('Evidence access and reproducibility','ReportH2'),
        par(f'The repository is public: <link href="{REPO}">{REPO}</link>. Source links below are pinned to evidence commit {SOURCE_COMMIT[:7]}. On this branch the Git root is the local PN2 directory; GitHub paths therefore begin with docs/, not PN2/docs/.','ReportSmall'),
        par('Local root: C:/Users/Seywe/Documents/GitHub/PointNeuron/PN2/<br/>Every source path below is relative to that root. The published report includes its numerical CSVs, figures and a source-hash manifest. Large checkpoints, caches and full local outputs under tmp/ are ignored and are not downloadable from GitHub.','ReportSmall'),
    ]
    for sid,(label,path) in SOURCE_FILES.items():
        # Deliberate line break between human-readable title and path keeps long paths readable.
        appendix.append(par(f'<b>{sid} {label}</b><br/><link href="{PIN+path}" color="#25445c">{html.escape(path)}</link>','ReportSmall'))
    appendix += [
        par('Portable report package','ReportH2'),
        par('docs/reports/progress_20261107/README.md<br/>development_results.csv and development_summary.csv<br/>source_manifest.json and figures/<br/>PointNeuron2_Progress_Report_Dr_Arij_20261107.pdf','ReportSmall'),
        par('The original design and earlier academic report are retained in the main checkout under docs/Proposal.pdf and output/pdf/PointNeuron_Proposal_Revised_Dr_Arij_2026-10-08.pdf. They are historical context, not a source of the fresh PN2 scores.','ReportSmall'),
    ]
    page('Appendix source records and access',appendix)

    pdf_path=OUT/'PointNeuron2_Progress_Report_Dr_Arij_20261107.pdf'
    def decoration(c,doc):
        c.saveState();c.setFont('Helvetica',8);c.setFillColor(GRAY)
        c.drawString(47,24,'PointNeuron 2 | Research progress | 7 November 2026')
        c.drawRightString(A4[0]-47,24,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(pdf_path),pagesize=A4,leftMargin=47,rightMargin=48,
                          topMargin=42,bottomMargin=42,title='PointNeuron 2 Research Progress Report',
                          author='',subject='Progress and research decisions for Dr Arij')
    story=[]
    for i,p in enumerate(pages):
        if i:story.append(PageBreak())
        story.extend(p)
    doc.build(story,onFirstPage=decoration,onLaterPages=decoration)

    # The same report text and numeric rows are editable and readable on GitHub.
    md=['# PointNeuron 2 research progress report','','Prepared for Dr Arij | 7 November 2026','',
        '[Download the PDF](PointNeuron2_Progress_Report_Dr_Arij_20261107.pdf) | [All development rows](development_results.csv) | [Unrounded summaries](development_summary.csv)','']
    for page_i,p in enumerate(pages):
        for flow in p:
            if isinstance(flow,Paragraph):
                if flow.style.name=='ReportTitle':
                    continue
                if page_i==0 and flow.text.startswith('Prepared for Dr Arij'):
                    continue
                text=flow.text
                text=text.replace('<br/>','\n').replace('<b>','**').replace('</b>','**').replace('<i>','*').replace('</i>','*')
                import re
                text=re.sub(r'<link href="([^"]+)"(?: color="[^"]+")?>(.*?)</link>',r'[\2](\1)',text)
                text=html.unescape(text)
                prefix='## ' if flow.style.name=='ReportH1' else '### ' if flow.style.name=='ReportH2' else ''
                if page_i==0 and prefix=='## ':prefix=''
                md.extend([prefix+text,''])
            elif isinstance(flow,Image):
                path=Path(flow.filename)
                md.extend([f'![{path.stem.replace("_"," ")}](figures/{path.name})',''])
            elif isinstance(flow,Table):
                data=flow._cellvalues
                def plain(v):
                    if isinstance(v,(tuple,list)):
                        return ' '.join(plain(x) for x in v)
                    t=v.text if isinstance(v,Paragraph) else str(v)
                    return html.unescape(t).replace('<br/>',' ').replace('|','/')
                rows2=[[plain(v) for v in row] for row in data]
                md.append('| '+' | '.join(rows2[0])+' |')
                md.append('| '+' | '.join(['---']*len(rows2[0]))+' |')
                md.extend('| '+' | '.join(row)+' |' for row in rows2[1:]);md.append('')
    md.extend(['## Full resolution saved review sheets','',
               '[WED SLP original five panel sheet](figures/alias020_XY_original.png)',
               '[Adult zebrafish original five panel sheet](figures/alias093_XY_original.png)',''])
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    manifest={'date':'2026-11-07','evidence_commit':SOURCE_COMMIT,'public_repository':REPO,
              'aggregation':'F1 pooled within neuron then seven-neuron macro; aggregate length pooled',
              'metrics_from_saved_reports_only':True,'new_scoring_or_model_execution':False,
              'Stage2_STOP_sha256':sha(ROOT/'tmp/stage2_trees_20261105/STOP.json'),
              'reserved_source_data_opened':False,
              'sources':{sid:{'label':label,'path':path,'sha256':sha(ROOT/path),'github_url':PIN+path} for sid,(label,path) in SOURCE_FILES.items()},
              'review_source_images':{str(source.relative_to(ROOT)):{'sha256':sha(source),'published_copy':str((FIG/(source.parent.name+'_XY_original.png')).relative_to(ROOT))} for source,_,_ in examples},
              'macro_values':aggregate,'oracle_values':oracle,
              'build_script':'scripts/build_progress_report_20261107.py',
              'build_script_sha256':sha(Path(__file__)),
              'validation':{'development_rows':len(detail),'stored_macro_F1_replay':'exact within 1e-14','all_seven_neurons_retained':True},
              'scope':'Report generation and saved evidence reformatting only; no protocol changed'}
    assert manifest['Stage2_STOP_sha256']=='0edf832a39d52bd5cd48296dbb5d613b1af51c0bdd5ea762f1af797425de6eae'
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (TMP/'build_result.json').write_text(json.dumps({'pdf':str(pdf_path),'planned_pages':len(pages),'rows':len(detail),'sources':len(SOURCE_FILES)},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pdf':str(pdf_path),'planned_pages':len(pages),'rows':len(detail),'sources':len(SOURCE_FILES)}))


if __name__=='__main__':
    main()
