"""Compare completed runs only on verified common fields/reference/ROI definitions.

This does not train missing baselines or certify equal compute/annotation quality.
An exploratory override permits unknown training provenance, not different datasets.
"""
from pathlib import Path
import json
from .io import read_csv,write_csv,write_json,fresh_output,finite
from .statistics import summarize,cluster_bootstrap


def compare_runs(run_dirs: list[Path], output: Path, allow_exploratory: bool=False) -> dict:
    if len(run_dirs)<2:
        raise ValueError('At least two completed evaluation run directories are required.')
    loaded=[]
    for directory in run_dirs:
        meta=json.loads((directory/'run_metadata.json').read_text(encoding='utf-8'))
        if meta['command']!='evaluate':
            raise ValueError('Only completed evaluation runs may be compared.')
        rows=read_csv(directory/'per_field_metrics.csv')
        if not rows or len({r['sample_id'] for r in rows})!=len(rows):
            raise ValueError('Empty or duplicate evaluation fields.')
        if any(r.get("model_id") != meta["model_id"] or r.get("run_id") != meta["run_id"] for r in rows):
            raise ValueError("Per-field model/run identity differs from run metadata.")
        converted=[]
        for r in rows:
            item={}
            for k,v in r.items():
                if v=='':item[k]=None
                elif k in ('sample_id','group_id','model_id','run_id','evaluation_role','provenance_status','roi_origin'):item[k]=v
                elif finite(v):item[k]=float(v)
                else:item[k]=v
            converted.append(item)
        hashes={}
        for h in read_csv(directory/'input_hashes.csv'):
            if h['input_role'] in ('roi_path','reference_labels_path','reference_boundary_path'):
                hashes[(h['sample_id'],h['input_role'])]=h['sha256']
        identity={r['sample_id']:(r['group_id'],r['fold'],r.get('um_per_pixel')) for r in converted}
        keys=('measurement_protocol','reference_postprocessing','iou_thresholds')
        protocol={k:meta['config'][k] for k in keys}
        independent=meta['provenance_statuses']==['checkpoint_verified'] and meta['evaluation_roles']==['held_out']
        if not independent and not meta['synthetic_only'] and not allow_exploratory:
            raise ValueError('Unverified/external comparisons require --allow-exploratory and cannot support a fair ranking claim.')
        loaded.append(dict(directory=directory,meta=meta,rows=converted,hashes=hashes,identity=identity,protocol=protocol))
    first=loaded[0]
    for run in loaded[1:]:
        if run['identity']!=first['identity']:
            raise ValueError('Runs differ in fields, specimen IDs, folds, or physical calibration. No silent intersection/drop is allowed.')
        if run['hashes']!=first['hashes']:
            raise ValueError('Exact reference/ROI inputs differ across runs. Harmonize them before comparison.')
        if run['protocol']!=first['protocol']:
            raise ValueError('Measurement/reference/IoU protocols differ across runs.')
        if run['meta']['evaluation_roles']!=first['meta']['evaluation_roles']:
            raise ValueError('Different evaluation roles cannot be mixed.')
    identifiers=[(r['meta']['model_id'],r['meta']['run_id']) for r in loaded]
    if len(set(identifiers))!=len(identifiers):
        raise ValueError('Duplicate model_id/run_id; each configuration needs a unique identity.')
    if len({r['meta']['synthetic_only'] for r in loaded})!=1:
        raise ValueError('Synthetic and real-source runs cannot be mixed.')
    common=set.intersection(*[{r['sample_id'] for r in run['rows'] if finite(r.get('delta_g'))} for run in loaded])
    fresh_output(output)
    summaries=[];ci={}
    for run in loaded:
        m=run['meta']
        for name,rows in [('all_fields_available_G_pairs',run['rows']),('common_G_defined_all_models',[r for r in run['rows'] if r['sample_id'] in common])]:
            summaries.append(dict(model_id=m['model_id'],run_id=m['run_id'],cohort=name,postprocessing_selection_status=m['config']['postprocessing_selection_status'],**summarize(rows)))
        ci[f"{m['model_id']}::{m['run_id']}"]=cluster_bootstrap(run['rows'],m['config']['bootstrap_replicates'],m['config']['bootstrap_seed'])
    write_csv(output/'model_comparison.csv',summaries)
    write_json(output/'per_model_cluster_ci.json',ci)
    result={'n_model_configurations':len(loaded),'n_all_fields':len(first['rows']),'n_common_g_fields':len(common),
            'common_g_sample_ids':sorted(common),'synthetic_only':first['meta']['synthetic_only'],
            'exploratory_override':allow_exploratory,
            'limitations':'Same evaluation inputs/protocol checked. Equal compute, checkpoint-selection fairness across model implementations, seed uncertainty and physical ground truth are not certified. Do not rank models only by G.'}
    write_json(output/'comparison_metadata.json',result)
    (output/'README.md').write_text('# '+('SYNTHETIC ONLY — ' if result['synthetic_only'] else '')+'Model-configuration comparison\n\n'+
        'Both all-field/available-pair and common-G-cohort results are provided. Failed fields remain in the first table.\n'+
        result['limitations']+'\n',encoding='utf-8')
    return result
