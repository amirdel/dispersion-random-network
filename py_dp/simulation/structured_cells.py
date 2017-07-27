import numpy as np

class structured_cells(object):
    def __init__(self,m,n,dx,dy,dz):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.m = m
        self.n = n
        self.nr_p = m*n
        self.x = np.zeros(self.nr_p)
        self.y = np.zeros(self.nr_p)
        #self.z = np.zeros(nr_p)
        self.create_geometry()
        self.p_n = np.zeros(self.nr_p)

    def indexToNum(self,i,j,m):
        """
        i is the row number, j is the column number, starting from 1
        value returned is the pore number starting from 0 and running 
        to nr_p-1
        """
        return ((j-1)*m + i -1)

    def create_geometry(self):
        m = self.m
        n = self.n
        xVal = 0.0
        xArray = self.x
        yArray = self.y
        dx = self.dx
        dy = self.dy
        for j in (np.array(range(n)) + 1):
            yVal = 0.0
            for i in (np.array(range(m)) + 1):
                pn = self.indexToNum(i, j, m)
                xArray[pn] = xVal
                yArray[pn] = yVal
                yVal -= dy
            xVal += dx
        xArray -= np.amin(xArray)
        yArray -= np.amin(yArray)
        xArray += dx / 2
        yArray += dy / 2
        return xArray, yArray

