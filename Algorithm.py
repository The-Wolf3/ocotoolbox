import numpy as np
import cvxpy as cp
import json

import matplotlib.pyplot as plt

class Algorithm:

    def __init__(self, n):
        self.n = n
        self.x = np.zeros((n,1))
        self.name = ""

    def __str__(self):
        return self.name

    def update(self):
        pass
    
    def setX(self, x0):
        self.x = x0.reshape((self.n, 1))

    def getX(self):
        return list(self.x.reshape(self.n))

    def eval(self, prob):
        return prob.f(self.x)

    def __iadd__(self, other):
        if isinstance(other, float):
            other = int(other)
        if isinstance(other, int):
            for _ in range(other):
                self.update()

class MOSP(Algorithm):

    def __init__(self, n, alpha=0.3, omega=2):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((2*n, 1))
        self.name = "MOSP"

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((np.vstack((A, -A)),
                            np.zeros((2*(self.n-A.shape[0]), A.shape[1]))))
        bigb = np.vstack((np.vstack((b, -b)),
                            np.zeros((2*(self.n-b.shape[0]), 1))))
        updateVec = self.om*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        grad = prob.grad(self.x)
        self.x += -self.alpha*(grad+bigA.T@self.dual)
        return self.x

class MOSPBasic(Algorithm):
    def __init__(self, n, alpha=0.03, omega=1.7):
        super().__init__(n)
        self.alpha = alpha
        self.om = omega
        self.dual = np.zeros((n, 1))
        self.T = 1
        self.name = "MOSP"

    def setDual(self, dual):
        self.dual = dual

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)

    def update(self, prob):
        self.T += 1
        alpha = self.alpha*self.T**(-1/3)
        omega = self.om*self.T**(-1/3)
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((A, np.zeros((self.n-A.shape[0], A.shape[1]))))
        bigb = np.vstack((b, np.zeros((self.n-b.shape[0], 1))))
        
        grad = prob.grad(self.x)
        self.x += -alpha*(grad+bigA.T@self.dual)

        updateVec = omega*(bigA@self.x-bigb)
        self.dual = np.maximum(self.dual+updateVec, 0)
        return self.x

class Lagrangian(Algorithm):

    def __init__(self, n, alpha=1, sigma=10):
        super().__init__(n)
        self.alpha = alpha
        self.sigma = sigma
        self.dual = np.zeros((n,1))
        self.name = "MALM"
        self.T = 1

    def update(self, prob):
        self.T += 1
        alpha = self.alpha*np.sqrt(self.T)
        sigma = self.sigma/np.sqrt(self.T) 
        A = prob.getA()
        b = prob.getb()
        bigA = np.vstack((A, np.zeros((self.n-A.shape[0], A.shape[1]))))
        bigb = np.vstack((b, np.zeros((self.n-b.shape[0], 1))))
        x = cp.Variable((self.n, 1))
        cost = 1/2/sigma*(cp.sum(cp.power(cp.pos(self.dual+sigma*(bigA@x-bigb)), 2))
                -np.sum(self.dual**2))+alpha/2*cp.sum(cp.power(x-self.x, 2))
        if isinstance(prob, ConvNetFlow):
            cost = cost + cp.sum(cp.multiply(prob.alpha, cp.power(x, 2))+cp.multiply(prob.beta, x)+prob.c)
        elif isinstance(prob, ExpNetFlow):
            cost = cost + cp.sum(cp.multiply(prob.alpha, cp.exp(cp.multiply(x, prob.beta)))+prob.c)
        convProb = cp.Problem(cp.Minimize(cost))
        convProb.solve()
        self.x = np.asarray(x.value).reshape((self.n, 1))
        self.dual = np.maximum(self.dual+self.sigma*(bigA@self.x-bigb), 0)


class OPENM(Algorithm):

    def __init__(self, n):
        super().__init__(n)
        self.name = "OPEN-M"

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        self.x += A.T@np.linalg.inv(A@A.T)@(b-A@self.x)
        grad = prob.grad(self.x)
        H = prob.getH(self.x)
        D = np.vstack((np.hstack((H, np.transpose(A))), 
                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        augGrad = np.vstack((grad, np.zeros((A.shape[0],1))))
        delta = (np.linalg.inv(D)@augGrad)[:self.n]
        self.x -= delta
        return self.x

    def violation(self, prob):
        A = prob.getA()
        b = prob.getb()
        return np.linalg.norm(A@self.x-b)


class IPM(Algorithm):

    def __init__(self, n, nlam):
        super().__init__(n)
        self.nlam = nlam
        self.eta = 1
        self.name = "IPM"
        self.lamda = np.zeros((nlam, 1))
        self.damped = True

    def setEta(self, eta):
        self.eta = eta

    def damped(self, isdamped):
        self.damped = isdamped

    def update(self, prob):
        A = prob.getA()
        b = prob.getb()
        grad, H = prob.gradHess(self.x)
        D = np.vstack((np.hstack((H, np.transpose(A))), 
                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        grad[-1] += self.eta
        augGrad = np.vstack((grad, np.round(A@self.x-b, 8)))
        delta = np.linalg.solve(D, augGrad)
        norm = augGrad.T@delta
        if norm > 1 and self.damped:
            delta = delta/norm
        print(norm)
        self.x -= delta[:self.n]
        self.lamda = delta[self.n:]
        return self.x
    
class IPMDamped(IPM):

    def update2(self, prob):
        A = prob.getA()
        b = prob.getb()
        grad, H = prob.gradHess(self.x)
        D = np.vstack((np.hstack((H, np.transpose(A))), 
                np.hstack((A, np.zeros((A.shape[0], A.shape[0]))))))
        grad[-1] += self.eta
        augGrad = np.vstack((grad, np.round(A@self.x-b, 8)))
        delta = np.linalg.solve(D, augGrad)
        norm = augGrad.T@delta
        count = 0
        while(not prob.getSOCPBarr().isFeasible(self.x - delta[:self.n])):
            delta *= 0.8
            count += 1
        print(norm, count)
        self.x -= delta[:self.n]
        self.lamda = delta[self.n:]
        return self.x

    def etaUpdate(self, prob, beta=1.02):
        self.update2(prob)
        self.eta *= beta
        return self.update2(prob)




class Function:

    def __init__(self, n=1):
        self.n = n

    def f(self, x):
        pass
    def grad(self, x):
        pass
    def hess(self, x):
        pass
    def cvxpyFunc(self):
        pass


class LinFunction(Function):

    def __init__(self, func, n=1, a=1, b=0):
        self.a = a
        self.b = 0
        self.func = func(n)

    def f(self, x):
        return self.func.f(self.a*x+self.b)

    def grad(self, x):
        return self.a*self.func.grad(self.a*x+self.b)

    

class Problem:

    def __init__(self, f=None, grad=None):
        self.f = f
        self.grad = grad

    def increment(self):
        pass
    def optimal(self):
        pass
    def setF(self, f):
        self.f = f
    def setGrad(self, grad):
        self.grad = grad


class LinConsProblem(Problem):

    def __init__(self, n=1):
        super().__init__(self)
        self.n = n
        self.A = np.zeros((0, n))
        self.b = np.zeros((0, 1))
        self.baseB = self.b

    def getA(self):
        return self.A
    
    def getb(self):
        return self.b

    def setA(self, A):
        self.A = A

    def setb(self, b):
        self.b = b
        self.baseB = b

    def violation(self, x):
        return np.linalg.norm(self.A@x-self.b.T)

def sigmoid(x):
    return 1/(1+np.e**-(x))

def sigmoidGrad(x):
    return sigmoid(x)**2 *np.e**-(x)

def sigmoidHess(x):
    return np.e**(-x)*(2*np.e**(-x)*sigmoid(x)**3)-sigmoidGrad(x)

class NetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.scaling = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.zeros((n, 1))

    def setA(self, conjMatrix):
        pass

    def loss(self, x):
        return np.sum(self.scaling*
            (sigmoid(self.alpha*x+self.beta)+sigmoid(-self.alpha*x)))
    
    def gradFct(self, x):
        return self.alpha*(sigmoidGrad(self.alpha*x+self.beta)-sigmoidGrad(-self.alpha*x))

    def hessFct(self, x):
        return self.alpha**2*(sigmoidHess((self.alpha*x+self.beta)*np.identity(self.n))+
                sigmoidHess(-self.alpha*x*np.identity(self.n)))
    
    def getH(self, x):
        return self.hessFct(x)

class ConvNetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.c = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.zeros((n, 1))
        self.T = 1

    def setLossParams(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def loss(self, x):
        return np.sum(self.alpha*x**2+self.beta*x+self.c)
    
    def gradFct(self, x):
        return self.alpha*2*x+self.beta

    def hessFct(self, x):
        return np.identity(self.n) *2* self.alpha
    
    def getH(self, x):
        return self.hessFct(x)

    def optimal(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, x**2)+cp.multiply(self.beta,x)+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver="GUROBI")
        return prob.value, x.value

    def optProblem(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, x**2)+cp.multiply(self.beta,x)+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        return prob

    def randomIncrement(self):
        self.alpha = np.random.sample(self.alpha.shape)*5

    def subIncrement(self):
        self.T += 1
        self.alpha = np.ones(self.alpha.shape) + np.random.sample(self.alpha.shape)*10/self.T

    def increment(self):
        self.randomIncrement()


class ExpNetFlow(LinConsProblem):

    def __init__(self, n=1):
        super().__init__(n)
        self.f = self.loss
        self.grad = self.gradFct
        self.c = np.ones((n, 1))
        self.alpha = np.ones((n, 1))
        self.beta = np.ones((n, 1))/10
        self.T = 1

    def setLossParams(self, alpha, beta, c):
        self.alpha = alpha
        self.beta = beta
        self.c = c

    def loss(self, x):
        return np.sum(self.alpha*np.e**(self.beta*x)+self.c)
    
    def gradFct(self, x):
        return self.alpha*self.beta*np.e**(self.beta*x)

    def hessFct(self, x):
        return np.identity(self.n) * self.alpha*self.beta**2*np.e**(self.beta*x)
    
    def getH(self, x):
        return self.hessFct(x)

    def optimal(self):
        x = cp.Variable((self.n, 1))
        cost = cp.sum(cp.multiply(self.alpha, cp.exp(cp.multiply(x, self.beta)))+self.c)
        constraints = []
        for row in range(self.A.shape[0]):
            constraints.append(cp.sum(cp.multiply(self.A[row].reshape(self.n,1), x)) == self.b[row])
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        return prob.value, x.value

    def randomIncrement(self, scaling=100):
        self.alpha = np.random.sample(self.alpha.shape)*scaling
        self.beta = np.random.sample(self.beta.shape)/10+1/10

    def subIncrement(self, scaling=100):
        self.alpha = np.random.sample(self.alpha.shape)*scaling/np.sqrt((self.T+20)**1.75)+1
        self.beta = np.random.sample(self.beta.shape)/10/np.sqrt((self.T+20)**1.75)+1/10
        self.b = self.baseB+np.random.sample(self.baseB.shape)*scaling/np.sqrt((self.T+20)**1.75)
        self.T += 1

    def increment(self):
        self.subIncrement()



class QLCP(Problem):

    def __init__(self, n):
        super().__init__()
        self.n = n
        self.increment()

    def increment(self):
        Q = np.random.random((self.n, self.n))
        Q = (Q + Q.T)/2
        def f(x):
            return float(x.T@Q@x)
        def grad(x):
            return 2*Q@x
        self.f = f
        self.grad = grad
        self.H = Q
        m = np.random.randint(1,5)
        self.A = np.random.random((m, self.n))
        self.b = np.random.random((m, 1))
    
    def getA(self):
        return self.A
    def getb(self):
        return self.b
    def getH(self, x):
        return self.H



class Benchmark:

    def __init__(self, n, prob, *algo):
        self.n = n
        self.prob = prob(n)
        self._doInit(algo)

    def __init__(self, prob, *algo):
        self.n = prob.n
        self.prob = prob
        self._doInit(algo)
        

    def _doInit(self, algo):
        self.data = {}
        self.algo = []
        self.algoNames = []
        for alg in algo:
            algInst = alg(self.n)
            self.algoNames.append(str(algInst))
            self.algo.append(algInst)
            self.data[algInst.name] = {'regret':[0],'avgRegret':[0],
                                        'roundRegret':[0],'violation':[0]}
        self.T = 0
        self.Taxis = [0]
        self.optimal = [0]



    def increment(self, probIncrement=None, n=1):
        for _ in range(n):
            self.T += 1
            self.Taxis.append(self.T)
            self.optimal.append(self.prob.optimal()[0])
            for alg in self.algo:
                regret = np.abs(self.prob.optimal()[0]-alg.eval(self.prob))
                self.data[alg.name]['roundRegret'] += [regret]
                last = self.data[alg.name]['regret'][-1]
                self.data[alg.name]['regret'].append(last+regret)
                self.data[alg.name]['avgRegret'].append((last+regret)/self.T)
                last = self.data[alg.name]['violation'][-1]
                self.data[alg.name]['violation'].append(self.prob.violation(alg.getX())+last)
                alg.update(self.prob)

            if probIncrement is None:
                self.prob.increment()
            else:
                probIncrement()

    def __add__(self, n):
        if isinstance(n, int):
            for _ in range(n):
                self.increment()
        return self

    def __iadd__(self, n):
        if isinstance(n, int):
            for _ in range(n):
                self.increment()
        return self

    def dump(self, filename='bench.json'):
        with open(filename, 'w') as file:
            dct = {}
            dct['data'] = self.data
            dct['Taxis'] = self.Taxis
            dct['n'] = self.n
            dct['T'] = self.T
            dct['algorithms'] = self.algoNames
            for alg in self.algo:
                dct['data'][alg.name]['x'] = alg.getX()
            json.dump(dct, file)
            file.close()

    def load(self, filename='bench.json'):
        with open(filename,'r') as file:
            dct = json.load(file)
            self.data = dct['data']
            self.Taxis = dct['Taxis']
            self.T = dct['T']
            self.n = dct['n']
            self.algo = []
            self.algoNames = dct['algorithms']
            for alg in self.algoNames:
                klass = globals()[alg]
                algInst = klass(self.n)
                algInst.setX(np.asarray(self.data[alg]['x']))
                self.algo.append(algInst)
            file.close()

    def loadData(self, filename='bench.json'):
        with open(filename, 'r') as file:
            dct = json.load(file)
            self.data = dct['data']
            self.Taxis = dct['Taxis']
            self.T = dct['T']
            self.n = dct['n']
            self.algoNames = dct['algorithms']
            file.close()

    def plotRegret(self):
        for alg in self.algo:
            plt.plot(self.Taxis, self.data[alg.name]['regret'],linewidth=2.0,label=alg.name)

    def plotAvgRegret(self):
        for alg in self.algo:
            plt.plot(self.Taxis, self.data[alg.name]['avgRegret'],linewidth=2.0,label=alg.name)

    def plot(self, property):
        for alg in self.algo:
            if property in self.data[alg.name]:
                plt.plot(self.Taxis, self.data[alg.name][property],linewidth=2.0,label=alg.name)

    def plotOptimal(self):
        if len(self.Taxis)>0:
            plt.plot(self.Taxis[1:], self.optimal[1:])




            


        


