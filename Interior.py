
from Algorithm import *
import numpy as np
from math import sqrt


class Barrier:

    def __init__(self, n):
        self.n = n

    def __add__(self, other):
        if other.n == self.n:
            ans = BarrSum(self.n)
            ans.fls.append(self)
            if isinstance(other, BarrSum):
                for func in other.fls:
                    ans.fls.append(func)
            else:
                ans.fls.append(other)
            return ans
        return None

class LogBar(Barrier):

    def __init__(self, b, loc, le=True):
        super().__init__(len(b))
        self.b = b
        self.loc = loc
        self.sign = -1 if le else 1
        self.P = np.zeros((self.n, 1))
        for i in loc:
            self.P[i] = 1

    def setb(self, b):
        self.b = b

    def logf(self, x):
        ans = 0
        for i in self.loc:
            ans += -np.log(self.sign*(x[i]-self.b[i]))
        return ans
    
    def f(self, x):
        ans = 0
        for i in self.loc:
            ans += -1*self.sign*(x[i]-self.b[i])
        return ans
    
    def grad(self, x):
        try:
            return -1/(x-self.b)*self.P
        except:
            return np.zeros(x.shape)
    
    def hess(self, x):
        return np.identity(self.n)*1/(x-self.b)**2*self.P
    
    def isFeasible(self, x):
        for i in self.loc:
            if -1*self.sign*(x[i]-self.b[i]) > 0 :
                return False
        return True

    
class QuadBar(Barrier):

    def __init__(self, n, A, B):
        super().__init__(n)
        self.A = A
        self.B = B

    def f(self, x):
        return (x.T@self.A@x+self.B.T@x).flatten()

    def grad(self, x):
        val = x.T@self.A@x+self.B.T@x
        return -(2*self.A@x + self.B)/val
    
    def hess(self, x):
        val = x.T@self.A@x+self.B.T@x
        grad = 2*self.A@x + self.B
        return -1/val*2*self.A + grad@grad.T/val**2
    
    def isFeasible(self, x):
        return self.f(x) < 0
        

class BarrSum(Barrier):

    def __init__(self, n):
        super().__init__(n)
        self.fls = []

    def f(self, x):
        ans = 0
        for func in self.fls:
            ans += func.f(x)
        return ans

    
    def grad(self, x):
        ans = np.zeros((self.n, 1))
        for func in self.fls:
            ans += func.grad(x)
        return ans
    
    def hess(self, x):
        ans = np.zeros((self.n, self.n))
        for func in self.fls:
            ans += func.hess(x)
        return ans

    def __iadd__(self, other):
        if other.n == self.n:
            if isinstance(other, BarrSum):
                for func in other.fls:
                    self.fls.append(func)
            else:
                self.fls.append(other)

    def __add__(self, other):
        if other.n == self.n:
            ans = BarrSum(self.n)
            for func in self.fls:
                ans.fls.append(func)
            if isinstance(other, BarrSum):
                for func in other.fls:
                    ans.fls.append(func)
            else:
                ans.fls.append(other)
            return ans
        return None

    def isFeasible(self, x):
        for func in self.fls:
            if not func.isFeasible(x):
                return False
        return True




class Holder:

    def __init__(self, case=None):
        if case is not None:
            self.N = case.N
            self.adj = np.conjugate(case.adj)
            self.genData = case.genData
            self.vlim = case.getVlim()
            self.cost = case.getCost()
            self.loads = case.getLoadData()
            self.mva = case.mva
            self.getParams(case)
            self.setA()
            self.setBarrier()


    def getParams(self, case):
        self.numSlack = 1
        self.numWreal = 0
        self.numWimag = 0
        self.numGen = 0
        self.genMap = {}
        for i in range(self.N):
            if not all(case.genData[i] == 0):
                self.genMap[i] = self.numGen
                self.numGen += 1

        self.realMap = {}
        self.imagMap = {}
        count = self.numGen*2
        for i in range(self.N):
            self.realMap[(i, i)] = count
            count += 1
            self.numWreal += 1
            for j in range(i+1, self.N):
                if self.adj[i, j] != 0:
                    self.realMap[(i, j)] = count
                    count += 1
                    self.numWreal += 1
        for i in range(self.N):
            for j in range(i+1, self.N):
                if self.adj[i, j] != 0:
                    self.imagMap[(i,j)] = count
                    count += 1
                    self.numWimag += 1

        self.total = self.numSlack+count

    def getBox(self):
        locp = []
        locq = []
        pmin = np.zeros((self.total, 1))
        pmax = np.zeros((self.total, 1))
        qmin = np.zeros((self.total, 1))
        qmax = np.zeros((self.total, 1))
        for i in self.genMap:
            pos = self.genMap[i]
            locp.append(pos)
            locq.append(pos+self.numGen)
            line = self.genData[i]
            pmax[pos] = line[0]
            pmin[pos] = line[1]
            qmax[pos+self.numGen] = line[2]
            qmin[pos+self.numGen] = line[3]
        return LogBar(pmax, locp)+LogBar(pmin, locp, False)+LogBar(qmax, locq)+LogBar(qmin, locq, False)

    def getWbox(self):
        loc = []
        vmax = np.zeros((self.total, 1))
        vmin = np.zeros((self.total, 1))
        for i in range(self.N):
            pos = self.realMap[(i,i)]
            loc.append(pos)
            vmax[pos] = self.vlim[i, 0]**2*1.01
            vmin[pos] = self.vlim[i, 1]**2*0.99
        return LogBar(vmax, loc) + LogBar(vmin, loc, False)
    
    def getQuad(self):
        A = np.zeros((self.total, self.total))
        b = np.zeros((self.total, 1))
        for i in self.genMap:
            pos = self.genMap[i]
            A[pos, pos] = self.cost[i, 0]*self.mva**2
            b[pos] = self.cost[i, 1]*self.mva
        b[-1] = -1
        return QuadBar(self.total, A, b)
     
    def getPos(self):
        return LogBar(np.zeros((self.total, 1)), [self.total-1], False)

    def getSOCP(self):
        ans = BarrSum(self.total)
        for i in range(self.N):
            for j in range(i+1, self.N):
                if (self.adj[i, j] != 0):
                    A = np.zeros((self.total, self.total))
                    b = np.zeros((self.total, 1))
                    Wii = self.getWreal(i, i)
                    Wjj = self.getWreal(j, j)
                    Wijreal = self.getWreal(i, j)
                    Wijimag = self.getWimag(i, j)
                    A[Wijreal, Wijreal] = 1
                    A[Wijimag, Wijimag] = 1
                    A[Wii, Wjj] = -0.5
                    A[Wjj, Wii] = -0.5
                    ans = ans + QuadBar(self.total, A, b)
        return ans
    
    def getSOCPBarr(self):
        return self.barrier + self.getSOCP()

    
    def getS(self, x):
        S = np.zeros((self.N*2, self.N*2))
        for i in range(self.N):
            for j in range(self.N):
                S[i*2, j*2] = x[self.getWreal(i,j)]
                S[i*2+1, j*2+1] = x[self.getWreal(i,j)]
                if i!=j:
                    S[i*2, j*2+1] = -x[self.getWimag(i, j)]
                    S[i*2+1, j*2] = x[self.getWimag(i, j)]
        return S
                    

    def getS2(self, x):
        ans = 0
        start = 2*self.numGen
        end = self.total-1
        for i in range(start, end):
            ans += self.getF(i)*x[i]
        return ans

    def getSDPgradHess(self, x):
        S = self.getS2(x)
        Sinv = np.linalg.inv(S)
        grad = np.zeros((self.total, 1))
        hess = np.zeros((self.total, self.total))
        start = 2*self.numGen
        end = self.total-1
        for i in range(start, end):
            SFi = Sinv@self.getF(i)
            grad[i] = np.trace(SFi)
            for j in range(start, end):
                SFj = Sinv@self.getF(j)
                hess[i, j] = np.trace(SFi@SFj)
        return grad, hess

    def getF(self, index):
        if index < 2*self.numGen+self.numWreal:
            return self.getrealF(index)
        return self.getimagF(index)

    def getrealF(self, index):
        i, j = self.getij(index)
        F = np.zeros((self.N*2, self.N*2))
        if i == j:
            F[i, i] = 1
            F[i+self.N, i+self.N] = 1
        else:
            F[i, j] = 1
            F[i+self.N, j+self.N] = 1
            F[j, i] = 1
            F[j+self.N, i+self.N] = 1
        return F
    
    def getij(self, index):
        if index-self.numGen*2 < self.numWreal:
            for entry in self.realMap:
                if self.realMap[entry] == index:
                    return entry
        else:
            for entry in self.imagMap:
                if self.imagMap[entry] == index:
                    return entry
    
    def getimagF(self, index):
        i, j = self.getij(index)
        F = np.zeros((self.N*2, self.N*2))
        if i != j:
            F[i, j+self.N] = 1
            F[i+self.N, j] = -1
            F[j, i+self.N] = 1
            F[j+self.N, i] = -1
        return F


    def setA(self):
        A = np.zeros((2*self.N, self.total))
        for i in range(self.N):
            if i in self.genMap:
                A[i, self.genMap[i]] = 1
                A[self.N+i, self.numGen+self.genMap[i]] = 1
            for j in range(self.N):
                if self.adj[i, j] != 0:
                    print(i,j)
                    gam = self.adj[i, j]
                    iiW = self.getWreal(i,i)
                    iWreal = self.getWreal(i,j)
                    iWimag = self.getWimag(i,j)
                    A[i, iiW] += -np.real(gam)
                    A[i, iWreal] += np.real(gam)
                    
                    A[self.N+i, iiW] += -np.imag(gam)
                    A[self.N+i, iWreal] += np.imag(gam)
                    if i < j:
                        A[i, iWimag] += -np.imag(gam)
                        A[self.N+i, iWimag] += np.real(gam)
                    else:
                        A[i, iWimag] += np.imag(gam)
                        A[self.N+i, iWimag] += -np.real(gam)

        self.A = A
        return A
    
    def getA(self):
        return self.A
    
    def getb(self):
        b = np.zeros((self.N*2, 1))
        for i in range(self.N):
            b[i] = self.loads[i, 0]
            b[self.N+i] = self.loads[i, 1]
        return b
            
    def updateLoads(self, case):
        self.loads = case.getLoadData()


    def getWreal(self, i, j):
        if i > j:
            i, j = j, i
        if (i,j) in self.realMap:
            return self.realMap[(i,j)]
        return None
    
    def getWimag(self, i, j):
        if i> j:
            i, j = j, i
        if (i,j) in self.imagMap:
            return self.imagMap[(i,j)]
        return None
    
    def setBarrier(self):
        self.barrier = self.getBox()+self.getWbox()+self.getQuad()+self.getPos()
    
    def gradHess(self, x):
        sdpGrad, sdpHess = self.getSDPgradHess(x)
        sdpGrad += self.barrier.grad(x)
        sdpHess += self.barrier.hess(x)
        return sdpGrad, sdpHess


    def getX0(self, p, q, W, loss):
        x = np.zeros((self.total, 1))
        for i in self.genMap:
            x[self.genMap[i]] = p[i]
            x[self.genMap[i]+self.numGen] = q[i]
        for i in range(self.N):
            x[self.realMap[(i,i)]] = np.real(W[i,i])
            for j in range(i+1, self.N):
                if (i, j) in self.realMap:
                    x[self.realMap[(i, j)]] = np.real(W[i,j])
                    x[self.imagMap[(i, j)]] = np.imag(W[i,j])
        x[-1] = loss
        return x


class SOCPopf(Holder):

    def __init__(self, case):
        super().__init__(case)
        self.barr = self.getSOCPBarr()

    def gradHess(self, x):
        return self.barr.grad(x), self.barr.hess(x)





