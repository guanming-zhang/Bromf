import sys
import os
import matplotlib.pyplot as plt
import itertools
import subprocess
import json
import time
import numpy as np


class BatchJobManager:
    def __init__(self, input_config, comp_config,root_dir,rm = False):
        self.input_config = input_config
        self._input_config = input_config
        self.comp_config = comp_config
        self.root_dir = root_dir

        self.count = 0
        self.check_input()
        if os.path.isdir(root_dir) and rm:
            subprocess.run(['rm','-r',root_dir])
        subprocess.run(['mkdir',root_dir])
        
    def check_input(self):
        if type(self.input_config['n_save']) == str:
            raise RuntimeError("n_save should be a int ")
    #scan two parameters
    def scan_params(self,scan_dict,parent_param = None,n_repeat = 1):
        '''
        root_dir--root directory to store the data
        scan_dict--the dictionary for parameters e,g {"phi":[0.1,0.2],"N":[100,200],"batch_szie":[1,2]}
        n_repeat--runs n_repeat computaion(with the same input parameters but different initial configuration) 
        parent_param -- a new directory will be created for each value of the parent_param 
        '''
        if parent_param:
            parent_param_values = scan_dict[parent_param]
            scan_dict.pop(parent_param)
            for ppv in parent_param_values:
                ppv_str = self.parameters2str({parent_param:ppv})
                sub_dir_name = os.path.join(self.root_dir,ppv_str)
                subprocess.run(['mkdir',sub_dir_name])
                count = 0
                for run_id in range(n_repeat):
                    self.input_config["runID"] = run_id
                    for pv_dict in self.get_unzip_dict(scan_dict):
                        data_dir_name = "dir{}#".format(count) + self.parameters2str(pv_dict) + "#runID_"+str(run_id) 
                        subprocess.run(['mkdir',os.path.join(sub_dir_name,data_dir_name)])
                        self.input_config[parent_param] = ppv
                        self.update_input_config(pv_dict)
                        self.write_input(os.path.join(sub_dir_name,data_dir_name))
                        self.submit_config(os.path.join(sub_dir_name,data_dir_name))
                        self.restore_input_config()
                        count += 1
        else:
            count = 0
            for run_id in range(n_repeat):
                self.input_config["runID"] = run_id
                for pv_dict in self.get_unzip_dict(scan_dict):
                    data_dir_name = "dir{}#".format(count) + self.parameters2str(pv_dict) + "#runID_"+str(run_id) 
                    subprocess.run(['mkdir',os.path.join(self.root_dir,data_dir_name)])
                    self.update_input_config(pv_dict)
                    self.write_input(os.path.join(self.root_dir,data_dir_name))
                    self.submit_config(os.path.join(self.root_dir,data_dir_name))
                    self.restore_input_config()
                    count += 1
            
                    
    def parameters2str(self,pv_dict):
        rtn = ""
        for p in pv_dict.keys():
            if type(pv_dict[p]) == float or type(pv_dict[p]) == np.float64:
                rtn +="{}_{:.6E}#".format(p,pv_dict[p])
            else:
                rtn +="{}_{}#".format(p,pv_dict[p])
        return rtn[:-1]
        
        
    def get_unzip_dict(self,scan_dict):
        '''
        scan_dict = {'a':[a1,a2..an],'b':[b1,b2..bn]...}
                  where a and b are parameter names in input_config.keys()
        return unzipped dicts, {'a':a1,'b':b1 ...},{'a':a1,'b':b2 ...}.
        '''
        value_lists=  list(itertools.product(*scan_dict.values())) 
        param_name_list = list(scan_dict.keys())
        for v_list in value_lists:
            param_value_dict = {param_name_list[i]:v_list[i] for i in range(len(param_name_list))}
            yield param_value_dict

    def update_input_config(self,config):
        for k in config.keys():
            self.input_config[k] = config[k]

    def restore_input_config(self):
        self.input_config = self._input_config

    def write_input(self,loc):
        with open(os.path.join(loc,"input.json"),'w') as fs:
            fs.write(json.dumps(self.input_config,indent=4))
    
    def submit_config(self,args,nsleep = 0.05):
        with open('./submit_batch.INI', 'r') as file:
            fstring = file.read()
        for key in self.comp_config:
            fstring = fstring.replace(key, self.comp_config[key])
        
        if type(args) != str:
            raise NotImplementedError("args must be a string")
        fstring = fstring.replace("ARGS", args)
        with open('./submit_batch.sbatch', 'w') as file:
            file.write(fstring)
        subprocess.run(["sbatch", "submit_batch.sbatch"])
        time.sleep(nsleep)
        self.count += 1
        subprocess.run(["rm", "submit_batch.sbatch"])
    
input_config1 = {
  "range": [100.0,100.0],
  "npts": [256,256],
  "dt": 0.005,
  "time_scheme": "forward-Euler",
  "n_steps": 120000,
  "n_save": 500,
  "bc": "periodic",
  "iv": "random-normal",
  "iv_srho": 0.1,
  "N": 10000,
  "phi":0.1,
  "T": 0.0,
  "compression": 1,
  "rel_epsilon": 0.1,
  "R": 1.0,
  "Gamma_inv": 1.0,
  "corr": 0.0,
  "system": "SGD"  
}

input_config2 = {
  "range": [100.0,100.0],
  "npts": [256,256],
  "dt": 0.005,
  "time_scheme": "forward-Euler",
  "n_steps": 50000,
  "n_save": 500,
  "bc": "periodic",
  "iv": "random-normal",
  "iv_srho": 0.1,
  "N": 10000,
  "R": 1.0,
  "D": 0.01,
  "T": 0.0,
  "Gamma_inv": 1.0,
  "compression": 1,
  "rel_epsilon": 0.001
}

input_config3 = {
  "range": [100.0,100.0],
  "npts": [256,256],
  "dt": 2e-6,
  "time_scheme": "predictor-corrector",
  "n_steps": 80000,
  "n_save": 200,
  "bc": "periodic",
  "iv": "coarse-grain",
  "iv_rel_window": 4.0,
  "N":10000,
  "phi":0.1,
  "Gamma_inv": 1.0,
  "compression": 1
}

# comp_config = {
#     "NTHREADS":"1",
#     "CONDA_ENV":"julia_env",
#     "JULIA_EXE":"../bromf/calculation_intg.jl",
#     "PLOT_EXE1":"../bromf/plot.jl",
#     "PLOT_EXE2":"../bromf/plot_all.jl",
#     "MEMORY":"16GB",
#     "TIME":"20:00:00"
# }

comp_config = {
    "NTHREADS":"1",
    "CONDA_ENV":"julia_env",
    "JULIA_EXE":"/home/sa7483/julia-1.10.2/sig-julia/bromf/calculation_intg.jl",
    "PLOT_EXE1":"/home/sa7483/julia-1.10.2/sig-julia/bromf/plot.jl",
    "PLOT_EXE2":"/home/sa7483/julia-1.10.2/sig-julia/bromf/plot_all.jl",
    "MEMORY":"16GB",
    "TIME":"20:00:00"
}

root_dir = '/scratch/sa7483/sig-julia/scripts/testing'
subprocess.run(['mkdir',root_dir])

input_config1["n_steps"] = 1000
input_config1["n_save"] = 10
input_config1["R"] =  1.0
input_config1["iv_srho"] = 0.01
input_config1["dt"] =  0.01          #0.005
input_config1["iv"] = "random-normal"
# input_config1["N"] = 4000
# input_config1["rel_epsilon"] = 0.01
# input_config1["T"] = 0.1

# input_config1["npts"] = [512,512]
# input_config1["range"] = [100.0,100.0]

# input_config1["npts"] = [256,256]
# input_config1["range"] = [50.0,50.0]

# # #smaller value of R/dx
# input_config1["npts"] = [256,256]
# input_config1["range"] = [100.0,100.0]

input_config1["npts"] = [512,512]
input_config1["range"] = [200.0,200.0]

comp_config["NTHREADS"] = "8"
comp_config["MEMORY"] = "12GB"
comp_config["TIME"] = "23:00:00"

#RO
input_config1["system"] = "RO"
input_config1["Gamma_inv"] = 0.0  
scan_dict = {'N':[318309], 'rel_epsilon':[1.0], 'corr':[0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]}    


# #BRO
# input_config1["system"] = "BRO"
# input_config1["Gamma_inv"] = 1.0   
# scan_dict = {'N':[318309], 'rel_epsilon':[1.0], 'corr':[0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]}    


# # SGD
# input_config1["system"] = "SGD"
# input_config1["Gamma_inv"] = 1.0  
# # scan_dict = {'N':[318309], 'rel_epsilon':[1.0], 'corr':[0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]} 


jobs1 = BatchJobManager(input_config1,comp_config,root_dir)
# jobs1.scan_params(scan_dict,"phi",n_repeat= 1)
jobs1.scan_params(scan_dict,"N",n_repeat= 1)
