import numpy as np
import AuxOps as aops
import time
import copy

class CDL():
        
    
    def __init__(self, datain, dictsize):
        
        
        '''tmp, self. wl= zip(*zip(*zip(*zip(*zip(datain)))))
        #tmp, data5 = zip(*zip(*zip(*zip(*zip(*zip(*din)))))) -->this unzipping in combination with the following shallows the 3rd rdd..
        tmp11, self.wh = zip(*zip(*zip(*tmp)))
        self.datain_h, self.datain_l = zip(*zip(*zip(*tmp11)))
        
        self.wh = np.reshape(self.wh, (np.array(self.wh).shape[1], np.array(self.wh).shape[2]))
        self.wl = np.reshape(self.wl, (np.array(self.wl).shape[1], np.array(self.wl).shape[2]))
        '''
        
        
        self.datain_h = np.array(datain[0])
        self.datain_l = np.array(datain[1])
        #self.datain_h = np.array(datain_h)
        #self.datain_l = np.array(datain_l)
        #print(np.array(self.datain_h).shape[0])
        #self.n = np.array(datain[0]).shape
        
        self.wh = np.zeros((dictsize,1))#np.random.random((dictsize, 1))
        self.wl = np.zeros((dictsize,1))#np.random.random((dictsize, 1))
        
        self.p = np.zeros((dictsize, 1))
        self.q = np.zeros((dictsize, 1))
        
        self.y1 = np.zeros((dictsize, 1))#self.wh.shape)#np.zeros(self.wh.shape)
        self.y2 = np.zeros((dictsize, 1))#self.wh.shape)
        self.y3 = np.zeros((dictsize, 1))#self.wh.shape)
        #self.lam = lam
        self.dictsize = dictsize
        #self.cnt = 0;
        
        #print(np.array(self.datain_h).shape)
        #print(np.array(self.datain_l).shape)
        
    #def updateVals(self, wh, wl, y1, y2, y3, p, q):
 #   def __setitem__(self, wh,wl, y1, y2, y3, p, q):
    '''def __setitem__(self, wh, wl):    
        
        self.wh = wh 
        self.wl = wl
#        self.y1 = y1
#        self.y2 = y2
#        self.y3 = y3
#        self.q = q 
#        self.p = p
        
    '''
  
def updateCD(cdlin, dictin_ht, dictin_lt, dtdh, dtdl, c1,c2,c3,cnt):
   #time1 = time.time()
    '''
    if cnt==0:
        cdlin.y1 = np.zeros((cdlin.dictsize,1))
        cdlin.y2 = np.zeros((cdlin.dictsize,1))
        cdlin.y3 = np.zeros((cdlin.dictsize,1))
        cdlin.p = np.zeros((cdlin.dictsize,1))
        cdlin.q = np.zeros((cdlin.dictsize,1))       
    '''
    
    y11= np.squeeze(cdlin.y1)
    y22 = np.squeeze(cdlin.y2)
    
    y33 = np.squeeze(cdlin.y3)
    pp = np.squeeze(cdlin.p)
    
    qq = np.squeeze(cdlin.q)
    
        
                     
    datain_h = np.array(cdlin.datain_h)#datain[0]
    datain_l = np.array(cdlin.datain_l)#datain[1]
        
    wl = np.squeeze(cdlin.wl)
    wh = np.squeeze(cdlin.wh)
        
    
    #whl = aops.calcW(datain_h, datain_l, np.squeeze(dictin_ht),np.squeeze(dictin_lt), dtdh, dtdl, y11,y22,y33,pp,qq, wh, wl, c1, c2, c3)
        
    #print(cnt)    
    whl = aops.calcW(datain_h, datain_l, np.squeeze(dictin_ht),np.squeeze(dictin_lt), dtdh, dtdl, wh, wl, c1,c2,c3, y11,y22,y33,pp,qq, cnt,cdlin.dictsize)
    #print('wh shape:')
    #print(whl[0].shape)
  
    pp = aops.updThr(whl[0]-y11/c1)
    #print('wl shape:')
    #print(whl[1].shape)
    qq = aops.updThr(whl[1]-y22/c2)
    
    y11 = aops.updateY(y11, c1,  pp, whl[0])
    y22 = aops.updateY(y22, c1,  qq, whl[1])
    y33 = aops.updateY(y33, c3,  whl[0], whl[1])   
    #cdlin.cnt +=1
        
    cdlin.wh = whl[0].copy()
    cdlin.wl = whl[1].copy()#, y11, y22, y33, pp, qq)
        
        #cdlout.swh = self.calcOutProd(np.reshape(datain_h, (datain_h.shape[0],1)), np.reshape(whl[0], (1, whl[0].shape[0])))
        #cdlout.swl = self.calcOutProd(np.reshape(datain_l, (datain_l.shape[0],1)), np.reshape(whl[0], (1, whl[1].shape[0])))
        #free(cdlin)
        
    cdlin.y1 = y11.copy() 
    cdlin.y2 = y22.copy()
    cdlin.y3 = y33.copy()
    cdlin.p = pp.copy()
    cdlin.q = qq.copy()
        
    return cdlin
            
    #print('time elapsed:')
    #print(time.time() - time1)   
    #print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')    
    #return (whl[0], whl[1], y11,y22,y33,pp,qq)    
    #return cdlin#(datain_h, datain_l, whl[0],whl[1], newp,newq, newy1,newy2,newy3)#out
    
def startCDL(datain, dictsize):
    
    #data1 = np.array(datain[0])
    #data2 = datain[1]
    mycdl=CDL(datain, dictsize)
    
    return mycdl




def setValues(datain, dictsize):
        
    
    
    tmp, y3 = zip(*zip(*zip(*zip(*zip(*zip(*zip(datain)))))))
    y2 = tmp[0][1]
    tmp11= tmp[0][0]
    #print(tmp11)
    y1 = tmp11[1]
    #print(y1)
    tmp22 = tmp11[0]
    #print(tmp22)
    q = tmp22[1]
    tmp33 = tmp22[0]
    #print(tmp33)
    p = tmp33[1]
    tmp44 = tmp33[0]
    wl = tmp44[1]
    tmp55 = tmp44[0]
    wh = tmp55[1]
    tmp66 = tmp55[0]
    din_l = tmp66[1]
    din_h = tmp66[0]
    
    cdlout = CDL([din_h, din_l], dictsize)
    
    cdlout.datain_h = din_h
    cdlout.datain_l = din_l
    cdlout.wh = wh
    cdlout.wl = wl
    cdlout.p = p
    cdlout.q = q
    cdlout.y1 = y1
    cdlout.y2 = y2
    cdlout.y3 = y3
    
    return cdlout
    
    
def calcOutProd(ina, inb):
    #outer product:     
    return np.dot(ina, inb)
