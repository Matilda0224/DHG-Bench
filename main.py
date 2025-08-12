from lib_utils.exp_agent import ExpAgent
from lib_models.HNN.preprocessing import algo_preprocessing
from lib_dataset.data_base import HyperDataset
from lib_dataset.preprocessing import data_processing
from parameter_parser import parameter_parser,method_config,set_task_args
from lib_dataset.data_perturbation import perturbation

if __name__ == '__main__':

    args = parameter_parser() 
    args = method_config(args) 
    args = set_task_args(args) 

    data=HyperDataset(args) 

    if args.task_type == 'hg_cls':
        data = data.multi_hypergraphs 
    else:
        data = data_processing(args,data)
        data._initialization_()

        if args.is_perturbed:
            if isinstance(args.pert_p,str):
                args.pert_p = eval(args.pert_p) 
            if args.pert_mode not in ['spar_label','flip_label']:
                print('Robustness Perturbation for Structure and Feature')
                data = perturbation(data,mode=args.pert_mode,p=args.pert_p,masks=None)

        data = algo_preprocessing(data,args)
    
    agent = ExpAgent(args)
    agent.running(args.task_type,data)