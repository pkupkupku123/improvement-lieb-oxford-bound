"""Joint mu/channel weights and independent-radius optimization on Slurm.

A trust region moves after acceptance on a twice-denser independent LP.
Every output is NUMERICAL_ONLY; continuous certification is a later gate.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from numba import set_num_threads
from scipy.optimize import minimize
from model import Model,load_profile


def save(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def main(model_class=Model):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--stages',type=int,nargs='+',default=[880,1760]);ap.add_argument('--rounds',type=int,default=6)
    ap.add_argument('--iterations',type=int,default=25);ap.add_argument('--threads',type=int,default=16)
    ap.add_argument('--weight-step',type=float,default=.03);ap.add_argument('--radius-step',type=float,default=.008)
    ap.add_argument('--max-seconds',type=float,default=0)
    ap.add_argument('--minimum-log-gap',type=float,default=1e-4)
    args=ap.parse_args();set_num_threads(args.threads);started=time.monotonic()
    assert args.minimum_log_gap>0
    r,w,data=load_profile(args.input);d,m=r.shape;assert m==100 and d in (2,3,4,5,6)
    r=r/r[0,-1];out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    history=[];calls=0;accepted_rounds=0;last_termination=None;completed_stages=[];stage_stopping=[]
    provenance=dict(input=args.input,input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                    host=os.uname().nodename if hasattr(os,'uname') else os.environ.get('COMPUTERNAME'),
                    solver_class=model_class.__name__,
                    source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ('search.py','model.py','active_model.py') if Path(__file__).with_name(name).exists()})
    def record(model,x,value,details,status):
        rr,ww=model.unpack(x)
        return dict(format='independent-channel-lo-profile-v1',status=status,channels=d-1,shells_per_measure=m,
                    independent_radii=True,radii=rr.tolist(),weights=ww.tolist(),validation_constant=float(value),
                    validation_intervals=model.intervals,details=details,history=history,calls=calls,
                    accepted_rounds=accepted_rounds,completed_stages=completed_stages,stage_stopping=stage_stopping,seconds=time.monotonic()-started,
                    parameter_bounds=dict(min_radius=1e-4,max_channel_radius=4.,channel_weight_limit=16.,minimum_log_radius_gap=args.minimum_log_gap),
                    search_configuration=dict(stages=args.stages,rounds=args.rounds,iterations=args.iterations,
                                              weight_step=args.weight_step,radius_step=args.radius_step,max_seconds=args.max_seconds),
                    optimized=['mu_weights','mu_radii','all_channel_weights','all_channel_radii','majorant'],**provenance)
    for intervals in args.stages:
        model=model_class(d-1,m,intervals);validation=model_class(d-1,m,2*intervals)
        center=model.pack(r,w);current,valid_details=validation.evaluate(center,False)
        save(out,record(validation,center,current,valid_details,'NUMERICAL_ONLY_RUNNING'))
        cs,global_bounds=model.constraints(min_log_gap=args.minimum_log_gap);gb=np.asarray(global_bounds);weight_step=args.weight_step;radius_step=args.radius_step
        rejected=0;stopping_reason='round_limit'
        for round_index in range(args.rounds):
            dimension=d*m
            width=np.r_[weight_step*(1+2*np.abs(center[:dimension])),np.full(dimension-1,radius_step)]
            # Mu masses use a smaller trust scale than signed channel masses.
            width[:m]=np.minimum(width[:m],.5*weight_step)
            bounds=list(zip(np.maximum(gb[:,0],center-width),np.minimum(gb[:,1],center+width)))
            best=[float('inf'),center.copy()];round_calls=0
            def objective(x):
                nonlocal calls,round_calls
                C,g,details=model.evaluate(x);calls+=1;round_calls+=1
                feasible=max(abs(cs[0]['fun'](x)))<1e-7 and min(cs[1]['fun'](x))>=-1e-9
                feasible=feasible and np.min(x[:m])>=-1e-10
                if feasible and C<best[0]:
                    best[:]=[C,x.copy()]
                    if round_calls%10==1:
                        print(json.dumps(dict(event='search',channels=d-1,intervals=intervals,round=round_index,
                                              calls=calls,sampled_constant=C,seconds=time.monotonic()-started)),flush=True)
                return C,g
            result=minimize(objective,center,jac=True,method='SLSQP',constraints=cs,bounds=bounds,
                            options={'maxiter':args.iterations,'ftol':1e-10,'disp':False})
            if not np.isfinite(best[0]):raise RuntimeError('No feasible candidate from the joint search')
            candidate=best[1];checked,checked_details=validation.evaluate(candidate,False)
            improved=checked<current-2e-8
            movement=float(np.max(np.abs(candidate-center)/width))
            event=dict(stage=intervals,round=round_index,sampled_constant=float(best[0]),validation_constant=checked,
                       previous_validation=current,accepted=bool(improved),normalized_movement=movement,
                       optimizer_success=bool(result.success),optimizer_message=str(result.message),
                       optimizer_iterations=int(result.nit),calls=round_calls,seconds=time.monotonic()-started)
            history.append(event);print(json.dumps(dict(event='validated_round',**event)),flush=True)
            last_termination=event
            if improved:
                center=candidate;current=checked;valid_details=checked_details;accepted_rounds+=1;rejected=0
                weight_step=min(args.weight_step*2,weight_step*1.15);radius_step=min(args.radius_step*2,radius_step*1.15)
            else:
                rejected+=1;weight_step*=.4;radius_step*=.4
            save(out,record(validation,center,current,valid_details,'NUMERICAL_ONLY_RUNNING'))
            save(out.with_suffix('.history.json'),history)
            if rejected>=3:
                stopping_reason='three_dense_validation_rejections';break
            if result.success and improved and abs(event['previous_validation']-checked)<1e-7 and movement<.7:
                stopping_reason='small_successful_step';break
            if args.max_seconds and time.monotonic()-started>=args.max_seconds:
                stopping_reason='elapsed_time_limit';break
        r,w=model.unpack(center);completed_stages.append(intervals)
        stage_stopping.append(dict(intervals=intervals,reason=stopping_reason,rounds=round_index+1))
        if stopping_reason=='elapsed_time_limit':break
    final_model=validation;center=final_model.pack(r,w)
    final,details=final_model.evaluate(center,False,True)
    final_record=record(final_model,center,final,details,'NUMERICAL_ONLY_SEARCH_FINISHED')
    final_record.update(termination=last_termination,global_optimum_claimed=False,
                        parameter_bounds=dict(min_radius=1e-4,max_channel_radius=4.,channel_weight_limit=16.,minimum_log_radius_gap=args.minimum_log_gap))
    save(out,final_record)
    print(json.dumps(dict(event='finished',channels=d-1,validation_constant=final,validation_intervals=final_model.intervals,
                          accepted_rounds=accepted_rounds,calls=calls,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
