import scipy,time
import numpy as np,pandas as pd
from numpy.random import rand as rd

#定义方程组
def equations(t,y,NumSpecies,growth_rate,Sigma,interaction,dilution):
    eq=[]#放一个空的list来装方程的表达式f(y)，dyi/dt=f(y)
    for i in range(NumSpecies):
        temp=interaction[:,i]*y#第i个物种对其他所有物种的相互作用和丰度的乘积
        PosSum=np.sum(temp[temp>0])#将tempt其中大于零的拎出来，这是促进关系
        NegSum=np.sum(temp[temp<0])#将其中tempt小于零的拎出来，这是竞争关系
        eq.append(growth_rate[i]*y[i]*(1+NegSum+Sigma[i]*PosSum/(1+PosSum))-dilution*y[i])#f(y)，可参阅谷歌学术上GLV方程，稍微改动了一点
    return eq

#定义微分方程解码器
def GLV(timespan,initial,NumSpecies,growth_rate,Sigma,interaction,dilution):
    sol=scipy.integrate.solve_ivp(equations,
                                  t_span=timespan,
                                  y0=initial,
                                  method='RK45',
                                  args=(NumSpecies,growth_rate,Sigma,interaction,dilution),
                                  dense_output=True)#微分方程解码器
    return sol.sol(timespan[1])#返回我们想要的时间下的丰度

#定义循环下结果处理的函数
def euler(result,threshold):#该函数为里层循环的结果处理 #euclidean 

    for x in range(result.size):

        result['percentage'][x]=tuple(result['percentage'][x])#改变储存格式，方便计算，避免循环赋值

        for y in range(x,result.size):

            if np.sqrt(np.sum(np.square(np.array(result['percentage'][x])-np.array(result['percentage'][y]))))<threshold:#当2个物种得欧式距离小于阈值threshold的时候

                result['percentage'][y]=tuple(result['percentage'][x])#把这两个物种弄成一个

            else:

                result['percentage'][y]=tuple(result['percentage'][y])#这两个物种不动

    return result.drop_duplicates(subset=['percentage']).size, result.value_counts().tolist()/np.sum(result.value_counts().tolist())
           #返回2个值，一个是里层循环有多少种稳态情况，一个是每种稳态占比

def func(NumSpecies,timespan,repeat,threshold,showindex):

    start_time=time.time()

    result=pd.DataFrame(columns=['parameters','condition','percentage'])#空的pandas来装我们的结果

    for i in range(repeat[0]):#repeat[0]外层循环，意思是随机 生长速度、相互作用和稀释度

        Sigma=0.5*rd(NumSpecies)#人为随机给的一个参数，无特定意义
        #1 NumSpecies * outerloop

        growth_rate=0.3+0.5*rd()+0.4*rd()*rd(NumSpecies)#菌群生长速度，随机生成
        #2 NumSpecies * outerloop

        tempt=(-1)**(rd(NumSpecies,NumSpecies)<(1-0.2*rd()**2)).astype('float')#中间变量，主要为相互作用改变符号，返回-1（竞争）和1（促进），-1多一些
        # 0.2*rd()**2 + rd() - 1 < 0 ? -1 : 1

        tempt[tempt==1]=tempt[tempt==1]/np.random.randint(low=2, high=5, size=tempt[tempt==1].shape)#中间变量再加工，对返回1的值除以2到5之间的整数，让正值变小
        
        interaction=tempt*(0.4+1.3*(1-rd()**3)-0.4*rd()+2*0.4*rd()*rd(NumSpecies, NumSpecies))*(rd(NumSpecies, NumSpecies)<(1-0.2*rd()**2))
        #相互作用矩阵，tempt为符号，中间是相互作用矩阵，最后的符号判断(rd(NumSpecies,NumSpecies)<(1-0.2*rd()**2))是随机赋予随机个0，因为相互作用非全连接
        #3 NumSpecies * NumSpecies * outerloop

        for index in range(NumSpecies):interaction[index, index]=-1#将相互作用矩阵对角线全赋值-1，因为自身对自身的作用为-1

        dilution=0.1+0.1*rd()#稀释度随机赋值
        #4 1 * outerloop

        if i%showindex==0:
            
            print(i,'th Loop cumcost:',int((time.time()-start_time)/60),'minutes')#选着适当输出间隔showindex

        loop=pd.DataFrame(columns=['percentage'])#用空pandas来装结果

        for j in range(repeat[1]):#里层循环repeat[1]，意思是让初始值随便选，看在这么多随机里面有多少种单稳态和多稳态出现

            initialtemp=rd(NumSpecies)#无特定意义，为初始值铺垫

            initial=initialtemp/np.sum(initialtemp)#归一化
            #5

            loop.at[j,'percentage']=GLV(timespan,initial,NumSpecies,growth_rate,Sigma,interaction,dilution)#用微分方程解码器返回丰度值装到loop这个空的pandas里面
            
        result.at[i,'parameters']=np.concatenate((growth_rate,
                                                  Sigma,
                                                  interaction[~np.eye(interaction.shape[0],dtype=bool)].reshape(-1),
                                                  dilution),axis=None)#在外层循环下，将所需要的参数全部连接在一起
        
        result.at[i,'condition'], result.at[i,'percentage']=euler(loop, threshold)#将里层循环的值交给第3个定义的函数处理，返回稳态数和稳态占比

    #result.to_excel('NumSpecies{a}repeat{b}.xlsx'.format(a=NumSpecies,b=repeat[0]*repeat[1]))

    return result#返回自变量和因变量的结果

import multiprocessing 
from numba import jit

if __name__=='__main__':
    NumSpecies=10#物种数
    timespan=[0,2500]#时间跨度，2500是反复查看，为比较合适的值

    repeat=[200,500]#外层循环200次，里层循环500次

    threshold=0.1#计算里层循环下稳态的欧氏距离阈值
    showindex=1#在外层循环下，每间隔多少输出一次
    n_cores =multiprocessing.cpu_count()#开多个核来跑，让外层循环200*n_cores，扩大样本量
    
    # Create list of parameter sets
    parameters = []
    for i in range(n_cores):
        parameters.append((NumSpecies,timespan,repeat,threshold,showindex))
    
    # Create processes
    processes = []
    for params in parameters:
        process = multiprocessing.Process(target=func, args=params)
        process.start()
        processes.append(process)
    
    # Join processes and collect results
    process_results = []
    for process in processes:
        process.join()
        process_results.append(func(*params))
    
    # Concatenate all DataFrames
    final_df = pd.concat(process_results).reset_index(drop=True)
    
    final_df.to_excel('result.xlsx')