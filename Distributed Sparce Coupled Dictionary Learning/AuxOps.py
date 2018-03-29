import numpy as np

from numpy.linalg import inv
#from numpy.linalg import solve

from scipy.linalg import solve 








        
    
def updateY(previousY, c, op1, op2, maxbeta=1e+6, beta=0.01):
        
    return  previousY + min(maxbeta, beta*c)*(op1-op2)
    #return previousY+np.ones((previousY.shape))
    
    
def updThresholding(inputmat, lam = 0.1):
        
        
    ttmp1 = np.maximum(np.fabs(inputmat) - lam, 0.)
       
       
    return ttmp1*np.sign(ttmp1)
        #return ttmp1
        
    
def updThr(inputmat, lam=0.1):
        
    th = lam/2.
    k= 0
    ttt = np.random.random(inputmat.shape)
    #ttt = a - th [a > th]
    
    for aa in inputmat:
        #print('*************')
        #print(aa)
        #print('*************')
        if aa>th:
            ttt[k] = aa-th
        elif abs(aa) <= th:
            ttt[k] = 0.
        elif aa < (-1.)*th:
            ttt[k] = aa +th
            
            
        k +=1    
            
            
    return ttt


    
    #def calcW(self, cdlin, dictin_h,dictin_l, c1,c2,c3):
    
 #[y1 = np.zeros((512,1)), y2 = np.zeros((512,1)), y3 = np.zeros((512,1)), p = np.zeros((512,1)), q = np.zeros((512,1))]]
'''
def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate
'''
#@static_vars(y1 = np.zeros((512,1)), y2 = np.zeros((512,1)), y3 = np.zeros((512,1)), p = np.zeros((512,1)), q = np.zeros((512,1)), previousCnt = -1) 
def calcW(datain_h, datain_l, dictin_ht, dictin_lt, dtdh, dtdl, wh,wl, c1,c2,c3, y1,y2,y3,p,q, cnt, dictsize):#, dictsize,kk):
    
            
    '''if cnt==0:
        calcW.y1 = np.zeros((dictsize,1))
        calcW.y2 = np.zeros((dictsize,1))
        calcW.y3 = np.zeros((dictsize,1))
        calcW.p = np.zeros((dictsize,1))
        calcW.q = np.zeros((dictsize,1))
        
    print(calcW.previousCnt)
    
    y11 = np.squeeze(calcW.y1)
    
    y22 = np.squeeze(calcW.y2)
    
    y33 = np.squeeze(calcW.y3)
    
    pp = np.squeeze(calcW.p)
    
    qq = np.squeeze(calcW.q)
    

    print(pp)
    
    '''
    
           
    tmp2 = np.dot(dictin_ht, np.transpose(np.array(datain_h))) + (y1 - y3) + c1*p + c3*wl
    
    tmp22 = np.dot(dtdh, tmp2)#+y1-y3)
    
    ####low
                      
    # + (y2 + y3) #+ c2*self.q + c3*self.wh
        
    tmp4 = np.dot(dictin_lt, np.transpose(np.array(datain_l))) + (y2 - y3) + c2*q + c3*wh
    
    tmp44 = np.dot(dtdl, tmp4)#+y2+y3)
    
    #print((wh-np.squeeze(y1)/c1).shape)
    
    #if calcW.previousCnt != cnt:
    
    #updThr(tmp22-y11/c1)#self.updThresholding(cdlout.wh-y1)
    #updThr(tmp44-y22/c2)#self.updThresholding(cdlout.wl-y2)
        
    #y11 = updateY(y11, c1,  pp, tmp22)
    
    #y22 = updateY(y22, c2, qq, tmp44)
    #y33 = updateY(y33, c3, tmp22, tmp44)
    
    #calcW.p = pp
    #calcW.q = qq
    
    #calcW.y1 = y11
    #calcW.y2 = y22
    #calcW.y3 = y33
    
    #updateVals(tmp22, tmp44, c1,c2,c3)

    #calcW.previousCnt  = cnt
    #print(kk)
    return [tmp22, tmp44]
    
    
