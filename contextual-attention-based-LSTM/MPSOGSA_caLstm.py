## MPSOGSA code
import os, json
import torch
import numpy as np
from numpy import random as rnd
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import sys
#from pyts.image import RecurrencePlot
import matplotlib.pyplot as plt
import argparse  

def euclid_dist(x,y):
    temp = 0   
    for i,j in zip(x,y):
        temp += (i-j)**2
        final = np.sqrt(temp)
    return final
def MPSOGSA(dataset,max_iters,num_particles,NumSave,lr,resume,savepath):
    os.makedirs(savepath,exist_ok=True)
    np.seterr(divide='ignore', invalid='ignore')
    # %config InlineBackend.figure_format = 'retina'
    c1 = 2
    c2 = 2
    g0 = 1
    w1=2;     
    wMax=0.9            
    wMin=0.5              
    current_fitness = np.zeros((num_particles,1))
    gbest_score = float('inf')
    OldBest=float('inf')

    convergence = np.zeros(max_iters)
    alpha = 20
    epsilon = 1

    class Particle:
        pass

    #all particle initialized
    particles = []
    Max=[700,400,300]
    dim =len(Max)+1
    gbest = np.zeros((1,dim))
    for i in range(num_particles):
        p = Particle()
        params=[]
        for k in range(len(Max)):
            params.append(np.random.randint(Max[k]*0.5,Max[k]))
        params.append(np.random.uniform(1e-5,0.01))
        #  p.params better be array
        p.params=np.array(params)
        p.fitness = rnd.rand()
        p.velocity = 0.3*rnd.randn(dim)
        p.res_force = rnd.rand()
        p.acceleration = rnd.randn(dim)
        p.force = np.zeros(dim)
        p.id = i
        particles.append(p)

    #training 
    print('training begain:', dataset)
    for i in range(max_iters):
        if i % 10 == 0:
            print('iteration number:', i)
        # gravitational constant
        g = g0*np.exp((-alpha*i)/max_iters)
        # calculate mse
        cf = 0    
        for p in particles:
            fitness = 0
            for kk,param in enumerate(p.params[:-1]):
                if param<Max[kk]*0.5 or param>Max[kk]:
                    p.params[kk]=np.random.randint(int(Max[kk]*0.5),Max[kk])  
            if p.params[-1]<1e-5 or p.params[-1]>0.01:
                p.params[-1]=np.random.uniform(1e-5,0.01)
  
            string_pparams = [str(int) for int in p.params]
            bashCommand = "python train_IEMOCAP.py  --batch-size 1024 " + " --model_parameters " +",".join(string_pparams)
            os.system(bashCommand)
            with open("results/dialoguernn_iemocap_results.json","r") as fid:
                results_info=json.load(fid)
            fitness=results_info[0]["best_loss"]
            
    #         fitness = fitness/X.shape[0]
            OldFitness=fitness
            current_fitness[cf] = fitness
            cf += 1
            if gbest_score > fitness and OldBest>fitness:
                """hiddenState=hidden0.cpu().detach().numpy()
                #rp = RecurrencePlot()
                X_rp = rp.fit_transform(hiddenState)
                plt.figure(figsize=(6, 6))
                plt.imshow(X_rp[0], cmap='binary', origin='lower')
            #         plt.title('Recurrence Plot', fontsize=14)
                plt.savefig(savepath+'RecurrencePlots/'+'RecurrencePlots_'+dataset+str(round(fitness,NumSave))
                             +'_'+str(hiddensize)+'_'
                                       +'.png',dpi=600)
                plt.show()

               
                weightsName='reservoir.weight_hh'
                for name, param in named_parameters:
        #             print(name,param)
                    if name.startswith(weightsName):
        #                 set_trace()
                        torch.save(param,savepath+'weights'+str(round(fitness,6))+'.pt') """
                OldBest=gbest_score
                gbest_score = fitness
                gbest = p.params

        best_fit = min(current_fitness)
        worst_fit = max(current_fitness)

        for p in particles:
            p.mass = (current_fitness[particles.index(p)]-0.99*worst_fit)/(best_fit-worst_fit)

        for p in particles:
            p.mass = p.mass*5/sum([p.mass for p in particles])


        # gravitational force
        for p in particles:
            for x in particles[particles.index(p)+1:]:
                p.force = (g*(x.mass*p.mass)*(p.params- x.params))/(euclid_dist(p.params,x.params))

        # resultant force
        for p in particles:
            p.res_force = p.res_force+rnd.rand()*p.force

        # acceleration
        for p in particles:
            p.acc = p.res_force/p.mass

        w1 = wMin-(i*(wMax-wMin)/max_iters)

        # velocity
        for p in particles:
            p.velocity = w1*p.velocity+rnd.rand()*p.acceleration+rnd.rand()*(gbest-p.params)

        # position
        for p in particles:
            p.params = p.params + p.velocity

        convergence[i] = gbest_score
#     set_trace()  
    plt.figure(figsize=(6, 6))
    plt.plot(convergence)  
    plt.xlabel('Convergence')
    plt.ylabel('Error')
    plt.draw()
    plt.savefig(savepath+dataset+'_ConvergenceChanges.png',dpi=600)  

    sys.stdout.write('\rMPSOGSA is training ESN (Iteration = ' + str(i+1) + ', MSE = ' + str(gbest_score) + ')')
    sys.stdout.flush()
        # save results 
    FileName=dataset+'_BestParameters.csv'
    #newdata=[{"max_iters":max_iters},{"num_particles":num_particles},{"parameters":p.params},{"convergence":convergence}]
    newdata=[{"max_iters":max_iters},{"num_particles":num_particles}]

    PathFileName=os.path.join(savepath,FileName)
    with open(PathFileName,"w") as fid:
        json.dump(newdata,fid)


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--dataset',default='iemocap',type=str, help='Dataset to train')

parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')

parser.add_argument('--max_iters', type=int,default=50, help='Max iterations')

parser.add_argument('--num_particles', type=int, default=30, help='Number of particles')

parser.add_argument('--NumSave', type=int, default=8, help='Number to save')

parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')

parser.add_argument('--savepath', type=str, required=False, default='./results/',
                    help='Path to save results')

args = parser.parse_args()

if __name__ =="__main__":
    MPSOGSA(args.dataset,args.max_iters,args.num_particles,args.NumSave,args.lr,args.resume,args.savepath)

