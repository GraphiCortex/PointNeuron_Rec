"""Diagnostic label-oracle gate and tie-aware precision-recall integration."""
import numpy as np


def precision_recall_area(labels,scores):
    """Stepwise area (average precision) and trapezoidal PR area; ties grouped.

    Unknown labels are excluded. No threshold, sign or calibration is fitted.
    A neuron with no positives has undefined AP, not an invented zero.
    """
    y=np.asarray(labels);s=np.asarray(scores,float)
    if y.shape!=s.shape or y.ndim!=1 or not np.isin(y,[-1,0,1]).all() or not np.isfinite(s).all():
        raise ValueError('Invalid labels/ranking scores')
    known=y>=0;y=y[known];s=s[known];positive=int((y==1).sum())
    result={'known_edges':len(y),'positive_edges':positive,'prevalence':positive/len(y) if len(y) else None,
            'average_precision':None,'trapezoidal_auprc':None}
    if not positive:return result
    order=np.argsort(-s,kind='stable');s=s[order];y=y[order]
    end=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(y)[end];precision=tp/(end+1);recall=tp/positive
    result['average_precision']=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    result['trapezoidal_auprc']=float(np.sum(np.diff(np.r_[0.,recall])*(np.r_[1.,precision[:-1]]+precision)/2))
    return result


def label_oracle_gate(label_f1,stage0_f1):
    if not all(np.isfinite(x) and 0<=x<=1 for x in (label_f1,stage0_f1)):
        raise ValueError('Invalid F1')
    return {'passed':bool(label_f1>=stage0_f1-.03),'label_oracle_f1_2':float(label_f1),
            'stage0_development_geometric_oracle_f1_2':float(stage0_f1),
            'maximum_drop':.03,'required_f1_2':float(stage0_f1-.03),
            'drop':float(stage0_f1-label_f1),
            'cohort_warning':'460 training label oracle versus151 development geometric oracle; matched training geometric oracle reported separately',
            'on_failure':'stop before selector training; no label repair or threshold tuning'}
