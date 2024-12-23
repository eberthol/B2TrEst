import sys
import numpy as np
import pandas as pd
from scipy import constants

from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection

'''
'''

# convinience functions

def to_deg(rad):
    return rad * 180/np.pi

def compute_point_on_line(P1, P2, y):
    ## y = az + b
    if P1[0]==P2[0]:
        z = P1[0]
    else:
        a = ( P1[1] - P2[1] )/( P1[0] - P2[0])
        if a==0:
            print(f'a = 0: P1 = {P1}     P2 = {P2}, y = {y}')
        b =  P1[1] - a * P1[0]
        z = (y - b)/a
    return np.array([z, y])

def compute_point_on_line_signed(P1, P2, y):
    if y>=0:
        return compute_point_on_line(P1, P2, y)
    else:
        return compute_point_on_line([P1[0], -P1[1]], [P2[0], -P2[1]], y)

def angle_0_to_2pi(angle):
    while True:
        if angle>=0 and angle<2*np.pi:
            break
        elif angle < 0:
            angle += 2*np.pi
        elif angle >= 2*np.pi:
            angle -= 2*np.pi 
    return angle


class CDC:
    """Simplified model of the CDC
        all lengths are in cm
        the 9 superlayers (SL) are labelled from 0 to 8, 0 being the innermost SL
        the 56 layers are similarly labelled from 0 to 55

    INPUT: dic
        if an empty dictionary is given, then the model described in the TDR is used
        the user can also specify their own CDC model by modifying the attibutes using a dictionary


    Attributes
    ----------
    A to H: numpy arrays
        coordinates of the boundary points, in the (z,y) plane as defined in the TDR (C here corresponds to C2 in Fig. 6.2)
        the z positions of G and H are computed by the framework
    rho_min, rho_max: numpy arrays
        min (max) distance of the superlayers (SL) from the point of origin in the (x-y) plane
        rho_min (rho_max) correspdond to the begining (end) of a SL
        Index 0 in the array corresponds to the SL that is the most inner one
    nLayers: numpy arrays 
        number of layers per SL
    nCells: numpy arrays 
        number of cells per layer, in the (x-y) plane
        it is assumed that all layers in a given SL have the same number of cells
    SL: pandas dataframe
        dataframe containing information about each super layer: position, number of layers, etc.
    layers: pandas dataframe
        dataframe containing information about layers: ID, number of cells, size, size of a cell etc.


    Public Methods
    --------------
    insideCDC(x, y, z, verbose=False)
        bool
        determines if a point with coordinates (x, y, z) is within the CDC volume
    get_cellID(phi, cell_delta_phi)
        int
        returns the index of a given cell (index 0 corresponds to the cell that has the samllest phi angle)
        input: phi position and cell size (in phi) of the given layer

    zy_contour(fig, ax)
        plot the CDC boundaries in the (z-y) plane
    zy_layers(fig, ax)
        plot the CDC layers in the (z-y) plane
    xy_contour(fig, ax)
        plot the CDC boundaries in the (x-y) plane
    xy_cells(fig, ax)
        plot the CDC cells in the (x-y) plane
        REMARK: if there are a lot of cells this method will significanlty slow down the plotting function
    """
    
    def __init__(self, dictionary):
        
        self.unit = '[cm]'
        self.A, self.B = np.array([ -83.1,  108.2+15.0  ]),  np.array([  158.6,  108.2+15.0  ])     
        self.C, self.D = np.array([ -68.2,   37.14 ]),       np.array([  144.9,   43.80 ])      
        self.E, self.F = np.array([ -47.4,   24.95 ]),       np.array([   87.7,   24.95 ]) 
        self.G, self.H = np.array([ -999,    16.0  ]),       np.array([   -999,   16.0  ]) 
        # super-layers (SL) geomertry
        self.rho_min = np.array([16.80, 25.70, 36.52, 47.69, 58.41, 69.53, 80.25,  91.37, 102.09])
        self.rho_max = np.array([23.80, 34.80, 45.57, 56.69, 67.41, 78.53, 89.25, 100.37, 111.14])
        # number of layers in a given super-layer 
        self.nLayers = np.array([   8,    6,     6,     6,     6,     6,     6,      6,      6]) 
        # number of cells in a given layer
        self.nCells  = np.array([ 160,  160,   192,   224,   256,   288,   320,    352,    384]) 

        for attr in self.__dict__:
            if attr in dictionary.keys():
                self.__setattr__(attr, dictionary[attr])
                
        self.G = compute_point_on_line(P1=self.C,  P2=self.E, y=self.G[1])
        self.H = compute_point_on_line(P1=self.D,  P2=self.F, y=self.H[1])
        
        self.SL = self._superLayers() 
        self.layers = self._layer_organisation() 
        
    def _layer_organisation(self):
        Ntot_layers = self.nLayers.sum()
        cdc_layers = pd.DataFrame()
        cdc_layers['layer_id'] = [i for i in range( 0, Ntot_layers )]
        cdc_layers.index = [f'lay{i}'  for i in range(0, cdc_layers.shape[0])]
        cdc_layers['SL'] = 'SL0'
        layer_min = int(self.SL.loc['SL0'].nLayers)
        for i in range(1, self.SL.shape[0]):
            cdc_layers['SL'] = np.where(cdc_layers.layer_id.isin(np.arange(layer_min, layer_min+6, 1)), f'SL{i}', cdc_layers.SL)
            layer_min += int(self.SL.loc[f'SL{i}'].nLayers)
        cdc_layers['nCells']   = 0
        cdc_layers['rho_size'] = 0
        cdc_layers['rho_min']  = 0
        cdc_layers['rho_max']  = 0
        for i in range(0, self.SL.shape[0]):
            sl = f'SL{i}'
            cdc_layers['nCells'] = np.where(cdc_layers.SL==sl, int(self.SL.loc[sl].nCells), cdc_layers.nCells)
            rho_size = (self.SL.loc[sl].rho_max - self.SL.loc[sl].rho_min)/self.SL.loc[sl].nLayers # layer size in rho
            cdc_layers['rho_size'] = np.where(cdc_layers.SL==sl, rho_size, cdc_layers.rho_size)
            rho_min0 = self.SL.loc[sl].rho_min
            relative_index = cdc_layers[cdc_layers.SL==sl].layer_id.values[0]
            cdc_layers['rho_min'] = np.where(cdc_layers.SL==sl, rho_min0+(cdc_layers.layer_id - relative_index)*cdc_layers.rho_size, cdc_layers.rho_min)
            cdc_layers['rho_max'] = np.where(cdc_layers.SL==sl, rho_min0+(cdc_layers.layer_id - relative_index)*cdc_layers.rho_size + cdc_layers.rho_size, cdc_layers.rho_max)
        cdc_layers['cell_delta_phi']  = 2*np.pi/cdc_layers.nCells ## size of one cell in phi
        
        cdc_layers = self._add_zShape_info(cdc_layers)
            
        return cdc_layers
        
    def _layer_length_z(P1, P2):
        return abs(P2[0] - P1[0])
    
    def _superLayers(self):
        cdc_SL = pd.DataFrame()
        cdc_SL['rho_min'] = self.rho_min
        cdc_SL['rho_max'] = self.rho_max
        cdc_SL['nLayers'] = self.nLayers
        cdc_SL['nCells']  = self.nCells
        cdc_SL.index = [f'SL{i}'  for i in range(0, cdc_SL.shape[0])]
        cdc_SL['maxLayerID'] = cdc_SL.nLayers.cumsum()
        cdc_SL['minLayerID'] = cdc_SL.maxLayerID - cdc_SL.nLayers
        cdc_SL['layer_size_rho'] = (cdc_SL.rho_max - cdc_SL.rho_min) / cdc_SL.nLayers
        cdc_SL = self._add_zShape_info(cdc_SL)
        return cdc_SL
    
    def _add_zShape_info(self, df):
        ## p1, p2 are the points defining the lines in the TDR (used to get z_bwd and z_fwd)
        # z_bwd(fwd) is the x position in the bwd(fwd) direction of the (super)Layer 
        df['p1_bwd'] = 0
        df['p1_bwd'] = np.where(df.rho_max.between(self.G[1], self.E[1]), df.apply(lambda r: self.E, axis=1),  df.p1_bwd)
        df['p1_bwd'] = np.where(df.rho_max.between(self.E[1], self.C[1]), df.apply(lambda r: self.C, axis=1),  df.p1_bwd)
        df['p1_bwd'] = np.where(df.rho_max.between(self.C[1], self.A[1]), df.apply(lambda r: self.A, axis=1),  df.p1_bwd)
        df['p2_bwd'] = 0
        df['p2_bwd'] = np.where(df.rho_max.between(self.G[1], self.E[1]), df.apply(lambda r: self.G, axis=1),  df.p2_bwd)
        df['p2_bwd'] = np.where(df.rho_max.between(self.E[1], self.C[1]), df.apply(lambda r: self.E, axis=1),  df.p2_bwd)
        df['p2_bwd'] = np.where(df.rho_max.between(self.C[1], self.A[1]), df.apply(lambda r: self.C, axis=1),  df.p2_bwd)
        df['z_bwd'] = 0
        df['z_bwd'] = df.apply(lambda row: compute_point_on_line(P1 = row.p1_bwd, P2 = row.p2_bwd, y = row.rho_max)[0], axis=1)

        df['p1_fwd'] = 0
        df['p1_fwd'] = np.where(df.rho_max.between(self.H[1], self.F[1]), df.apply(lambda r: self.F, axis=1),  df.p1_fwd)
        df['p1_fwd'] = np.where(df.rho_max.between(self.F[1], self.D[1]), df.apply(lambda r: self.D, axis=1),  df.p1_fwd)
        df['p1_fwd'] = np.where(df.rho_max.between(self.D[1], self.B[1]), df.apply(lambda r: self.B, axis=1),  df.p1_fwd)
        df['p2_fwd'] = 0
        df['p2_fwd'] = np.where(df.rho_max.between(self.H[1], self.F[1]), df.apply(lambda r: self.H, axis=1),  df.p2_fwd)
        df['p2_fwd'] = np.where(df.rho_max.between(self.F[1], self.D[1]), df.apply(lambda r: self.F, axis=1),  df.p2_fwd)
        df['p2_fwd'] = np.where(df.rho_max.between(self.D[1], self.B[1]), df.apply(lambda r: self.D, axis=1),  df.p2_fwd)
        df['z_fwd'] = 0
        df['z_fwd'] = df.apply(lambda row: compute_point_on_line(P1 = row.p1_fwd, P2 = row.p2_fwd, y = row.rho_max)[0], axis=1)
        return df
                
    def __repr__(self):
        # what is displayed when writing the name of the cass in the command line
        return f"<CDC simplified model>"

    def __str__(self):
        # what is displayed when using the print() function
        return f"CDC simplified model\n  {cdc.__dict__}\n"

    def insideCDC(self, x, y, z, verbose=False):    
        inCDC = True
        rho = np.sqrt(x**2 + y**2)
        id_last_layer = self.layers.shape[0]-1
        zL, zR = 0, 0
        ## check in z
        if rho<=self.rho_min[0]: 
            # actually, the particle is not in CDC but between the beam pipe and CDC... å
            pass
        else:
            # compute the limits in z at a given y
            zL = compute_point_on_line_signed(self.C, self.E, rho)[0]  if abs(rho)<=self.C[1] else compute_point_on_line_signed(self.A, self.C, rho)[0]
            zR = compute_point_on_line_signed(self.F, self.D, rho)[0]  if abs(rho)<=self.D[1] else compute_point_on_line_signed(self.D, self.B, rho)[0] 
            if z<zL or z>zR:
                inCDC=False
                if verbose: print(f'particle is at (x,y,z) = ({x:.1f}, {y:.1f}, {z:.1f}), zL = {zL:.1f}, zR = {zR:.1f}, rho = {rho:.1f}, rhoM = {self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]:.1f}' )
                if verbose: print(f'  ==> particle exited in the z direction' )

        ## check in rho direction
        if  rho>self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]: 
            inCDC=False
            if verbose: print(f'particle is at (x,y,z) = ({x:.1f}, {y:.1f}, {z:.1f}), zL = {zL:.1f}, zR = {zR:.1f}, rho = {rho:.1f}, rhoM = {self.layers[self.layers.layer_id==id_last_layer].rho_max.values[0]:.1f}' )
            if verbose: print(f' ==> particle exited in the  rho direction')

        return inCDC

    def get_cellID(self, phi, cell_delta_phi):
        if cell_delta_phi > 0:
            cell_id = int(np.floor(angle_0_to_2pi(phi)/cell_delta_phi))
        else:
            cell_id = -1
        return cell_id

    # plots
    def zy_contour(self, fig, ax):
         ## upper-half (y>0)
        ax.plot( [self.A[0],  self.B[0]],   [self.A[1],  self.B[1]]  )  ## AB
        ax.plot( [self.D[0],  self.B[0]],   [self.D[1],  self.B[1]]  )  ## DB
        ax.plot( [self.A[0],  self.C[0]],   [self.A[1],  self.C[1]]  )  ## AC
        ax.plot( [self.C[0],  self.E[0]],   [self.C[1],  self.E[1]]  )  ## CE
        ax.plot( [self.F[0],  self.D[0]],   [self.F[1],  self.D[1]]  )  ## FD
        ax.plot( [self.E[0],  self.G[0]],   [self.E[1],  self.G[1]]  )  ## EG
        ax.plot( [self.F[0],  self.H[0]],   [self.F[1],  self.H[1]]  )  ## FH
        ax.plot( [self.G[0],  self.H[0]],   [self.G[1],  self.H[1]]  )  ## GH

        ## lower-half (y<0)
        ax.plot( [self.A[0],  self.B[0]],   [-self.A[1],  -self.B[1]]  )  ## AB
        ax.plot( [self.D[0],  self.B[0]],   [-self.D[1],  -self.B[1]]  )  ## DB
        ax.plot( [self.A[0],  self.C[0]],   [-self.A[1],  -self.C[1]]  )  ## AC
        ax.plot( [self.C[0],  self.E[0]],   [-self.C[1],  -self.E[1]]  )  ## CE
        ax.plot( [self.F[0],  self.D[0]],   [-self.F[1],  -self.D[1]]  )  ## FD
        ax.plot( [self.E[0],  self.G[0]],   [-self.E[1],  -self.G[1]]  )  ## EG
        ax.plot( [self.F[0],  self.H[0]],   [-self.F[1],  -self.H[1]]  )  ## FH
        ax.plot( [self.G[0],  self.H[0]],   [-self.G[1],  -self.H[1]]  )  ## GH
        
        ax.set_xlabel(f'z {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)

    def zy_layers(self, fig, ax):
        # Rectangle(xy, width, height, *, angle=0.0, rotation_point='xy', **kwargs)
        #       +------------------+
        # :     |                  |
        # :   height               |
        # :     |                  |
        # :    (xy)---- width -----+
        patches  = [ Rectangle((self.SL.loc[sl].z_bwd,  self.SL.loc[sl].rho_min), 
                            width=self.SL.loc[sl].z_fwd - self.SL.loc[sl].z_bwd, 
                            height=(self.SL.loc[sl].rho_max - self.SL.loc[sl].rho_min) ) for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])]]
        patches += [ Rectangle((self.SL.loc[sl].z_bwd,  - self.SL.loc[sl].rho_max), 
                            width=self.SL.loc[sl].z_fwd - self.SL.loc[sl].z_bwd, 
                            height=(self.SL.loc[sl].rho_max-self.SL.loc[sl].rho_min) ) for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])]]
        p = PatchCollection(patches, alpha=0.0)
        ax.add_collection(p)
        
        patches  = [ Rectangle((self.layers.loc[lay].z_bwd,  self.layers.loc[lay].rho_min), 
                        width=self.layers.loc[lay].z_fwd - self.layers.loc[lay].z_bwd, 
                        height=(self.layers.loc[lay].rho_max - self.layers.loc[lay].rho_min) ) for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])]]
        patches += [ Rectangle((self.layers.loc[lay].z_bwd,  -self.layers.loc[lay].rho_max), 
                        width=self.layers.loc[lay].z_fwd - self.layers.loc[lay].z_bwd, 
                        height=(self.layers.loc[lay].rho_max - self.layers.loc[lay].rho_min) ) for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])]]
        p2 = PatchCollection(patches, alpha=0.4, linewidth=1, edgecolor='b')
        ax.add_collection(p2)
            
        
        ax.set_xlabel(f'z {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)
        
        ax.set_xlim(self.A[0], self.B[0])
        ax.set_ylim(-self.A[1], self.A[1])

    def xy_contour(self, fig, ax):
        # a wedge centered at x, y center with radius r that sweeps theta1 to theta2 (in degrees)
        # if width is given, then a partial wedge is drawn from inner radius r - width to outer radius r
        patches = [ Wedge((0, 0), 
                      self.SL.loc[sl].rho_max, 
                      0, 360, 
                      width=( self.SL.loc[sl].rho_max-self.SL.loc[sl].rho_min  ))  for sl in [f'SL{i}' for i in range(0, self.SL.shape[0])] ]
        ax.add_collection( PatchCollection(patches,  alpha=0.4, linewidth=2, edgecolor='b') )
        

        patches = [ Wedge((0, 0), 
                        self.layers.loc[lay].rho_max, 
                        0, 360, 
                        width=( self.layers.loc[lay].rho_max-self.layers.loc[lay].rho_min  ))  for lay in [f'lay{i}' for i in range(0, self.layers.shape[0])] ]
        ax.add_collection( PatchCollection(patches, alpha=0.4, linewidth=2, edgecolor='b', ls=':') )

        ax.set_xlim(-self.A[1], self.A[1])
        ax.set_ylim(-self.A[1], self.A[1])
        ax.set_xlabel(f'x {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)
        ax.set_aspect( 1 )

    def xy_cells(self, fig, ax):
        patches = []
        for l in range(0, self.layers.shape[0]):
            lay = f'lay{l}'
            patches += [ Wedge((0, 0), 
                            self.layers.loc[lay].rho_max, 
                            i*to_deg(self.layers.loc[lay].cell_delta_phi), i*to_deg(self.layers.loc[lay].cell_delta_phi)+to_deg(self.layers.loc[lay].cell_delta_phi), 
                            width=( self.layers.loc[lay].rho_max-self.layers.loc[lay].rho_min  )) for i in range(0, self.layers.loc[lay].nCells) ]
        
        ax.add_collection( PatchCollection(patches, linewidth=2, edgecolor='b', ls=':') )
        ax.set_xlim(-self.A[1], self.A[1])
        ax.set_ylim(-self.A[1], self.A[1])
        ax.set_xlabel(f'x {self.unit}', fontsize=20)
        ax.set_ylabel(f'y {self.unit}', fontsize=20)
        ax.set_aspect( 1 )
    
class Particle:
    """
    Given the charge, 3-momentum and production vertex of the particle, this class
    compute the information related to the point of closest approach (POCA)

    Attributes
    ----------
    q: int
        charge
    px, py, pz: float
        3-momentum of the particle
    prodVtxX, prodVtxY, prodVtxZ: float
        coordinates of the production vertex of the particle

    x0, y0, z0: float
        coordinates of the POCA 
    d0, phi0, omega, tanLambda: float
        helix parametrs
    s0: float
        arclength in the (x-y) plan at the POCA
        
    """
    def __init__(self, dictionary):
        self.q = None 
        self.px, self.py, self.pz = None, None, None 
        self.prodVtxX, self.prodVtxY, self.prodVtxZ =  None, None, None 
        
        self.x0, self.y0, self.z0 = None, None, None 
        self.d0, self.phi0, self.omega, self.tanLambda, self.s0 = None, None, None, None, None

        for attr in self.__dict__:
            if attr in dictionary.keys():
                self.__setattr__(attr, dictionary[attr])

        if self.x0 is None:
            self.x0, self.y0, self.z0, self.d0, self.phi0, self.omega, self.tanLambda, self.s0 = self._get_helix_paramters_at_POCA()
                
    def _get_helix_paramters_at_POCA(self):

        phi = np.arctan2(self.py, self.px)
        pt = np.sqrt(self.px**2 + self.py**2)
        
        # track curvature
        Bz = 1.5
        a = Bz/(constants.c*1e-6) # constants.c is in m.s-1
        omega     = a * self.q / pt   # [cm-1]   constants.c*1e-9 = 0.3 (which corresponds to cyclotron radius in m, we want cm)  
        tanLambda = self.pz/pt 

        Dpar  = -self.prodVtxX * np.cos(phi) - self.prodVtxY * np.sin(phi)
        Dperp = -self.prodVtxY * np.cos(phi) + self.prodVtxX * np.sin(phi)
        A = 2 * Dperp + omega * ( Dperp**2 + Dpar**2 )
        U = np.sqrt( 1 + omega * A )

        dphi = np.arctan2( omega * Dpar, 1 + omega * Dperp ) 
        s = dphi/omega 

        d0 = A / ( 1 + U )
        phi0 = phi - dphi # np.arctan2(py, px) - np.arctan2( omega * Dpar, 1 + omega * Dperp )
        z0 = self.prodVtxZ + s * tanLambda 

        x0 = d0 * np.sin(phi0) 
        y0 = -d0 * np.cos(phi0) 

        return x0, y0, z0, d0, phi0, omega, tanLambda, s 

class Point(Particle):
    """
    Get any point on the trajectory of a given particle and also computes its position in the CDC


    Attributes
    ----------
    instanceCDC: CDC
        CDC model
    inCDC: bool
        is the point in the CDC?
    layer_id: int
        CDC layer corresponding to the point
        the special value -1 means that no layer is associated to this point
    cell_id: int
        cell corresponding to the point
        the special value -1 means that no cell is associated to this point
    
    x, y, z, rho, phi, s: float
        coordinate, rho (vector from the origin the x-y plane), phi angle and arc-length corresponding to the point        
    
    """
    def __init__(self, dictionary, instanceCDC, s):
        super().__init__(dictionary)
        self.instanceCDC = instanceCDC 
        self.layer_id = -1 
        self.cell_id = -1 

        self.x, self.y, self.z, self.phi, self.s = self._get_particle_position(s)
        self.inCDC = instanceCDC.insideCDC(self.x, self.y, self.z, verbose=False) 
        self.rho =  np.sqrt(self.x**2 + self.y**2)

        self.azimutal_angle = np.arctan2(self.y, self.x)
        
        if self.inCDC:
            vals = instanceCDC.layers.loc[(instanceCDC.layers['rho_min'] <= self.rho) & (instanceCDC.layers['rho_max'] > self.rho)].layer_id.values
            if len(vals)==1:
                self.layer_id = instanceCDC.layers.loc[(instanceCDC.layers['rho_min'] <= self.rho) & (instanceCDC.layers['rho_max'] > self.rho)].layer_id.values[0]
                self.cell_id = instanceCDC.get_cellID(self.azimutal_angle, instanceCDC.layers.loc[f'lay{self.layer_id}'].cell_delta_phi)
            elif len(vals)>1:
                sys.exit('More than one CDC layer correspond to that point. That should not happend. Aborting.')

    def _get_particle_position(self, s):
        # returns particle position at a given arc length

        phi = self.phi0 - s*self.omega # not used in the computation

        z_prime = self.z0 + s * self.tanLambda

        Cs = self.omega * s
        x_prime = self.x0 + s * np.sin( 0.5 * Cs )/(0.5 * Cs) * np.cos(self.phi0 - 0.5 * Cs) if Cs!=0 else self.x0
        y_prime = self.y0 + s * np.sin( 0.5 * Cs )/(0.5 * Cs) * np.sin(self.phi0 - 0.5 * Cs) if Cs!=0 else self.y0

        return x_prime, y_prime, z_prime, phi, s
    
    def __repr__(self):
        return f"<Point at phi = {self.phi:.3f}>"

class Trajectory(Particle):
    """
        collection of points representing the trajectory of the particle
        computes the number of hits in the CDC
        the distance between the points (steps) is given as a function of the arc length (s)

    Attributes
    ----------
        signed_step: float
            for steps in arc length, this is equal to step_size
        steps: numpy array
            array containing the steps
        CDChits: int
            number of CDC hits 
        meanStepsPerCell_all, stdStepsPerCell_all, minStepsPerCell_all, maxStepsPerCell_all: float
            mean, standard deviation, min and max number on steps per cell
            can be used to tune the cut on the number of steps
        meanStepsPerCell_SL0, stdStepsPerCell_SL0, minStepsPerCell_SL0, maxStepsPerCell_SL0: float
            mean, standard deviation, min and max number on steps per cell for the dense superlayer (SL0)
            can be used to tune the cut on the number of steps
        meanStepsPerCell_SL1_SL8, stdStepsPerCell_SL1_SL8, minStepsPerCell_SL1_SL8, maxStepsPerCell_SL1_SL8: float
            mean, standard deviation, min and max number on steps per cell for the outer superlayer (SL1 to SL8)
            can be used to tune the cut on the number of steps
        min_setps_in_SL0_cell: int 
            minimum number of steps in a SL0 cell to be counted as a hit
        min_setps_in_outer_cell: int
            minimum number of steps in a SL1-8 cell to be counted as a hit
        list_of_points: list
            list of Point() objects forming the trajectory
        points: pandas dataframe
            for each point, provide the coordinates, the angle phi, the arc length s and the corresponding CDC layer and cell
        
    Public Methods
    --------------
        xy_trajectory(self, ax, label='', color='b')
            plot the trajectory in the (x-y) plane
        zy_trajectory(self, ax, label='', color='b')
            plot the trajectory in the (z-y) plane
        xy_vertex(self, ax, label='', color='b')
            plot the vertex position in the (x-y) plane
        zy_vertex(self, ax, label='', color='b')
            plot the vertex position in the (z-y) plane
        xy_poca(self, ax, label='', color='b')
            plot the POCA position in the (x-y) plane
        zy_poca(self, ax, label='', color='b')
            plot the POCA position in the (x-y) plane
            
    """
    def __init__(self, dictionary, instanceCDC, step_size, min_steps_in_SL0_cell=2, min_steps_in_outer_cell=10):
        super().__init__(dictionary)

        self.signed_step = None 
        self.steps = None

        self.signed_step = step_size
        self.steps = np.arange(self.s0, self.s0+1000, self.signed_step) # 1000 is a dummy value

        self.points = None 
        self.CDChits = None 

        self.min_steps_in_SL0_cell   = min_steps_in_SL0_cell 
        self.min_steps_in_outer_cell = min_steps_in_outer_cell 
        
        self.meanStepsPerCell_all, self.stdStepsPerCell_all, self.minStepsPerCell_all, self.maxStepsPerCell_all = None, None, None, None
        self.meanStepsPerCell_SL0, self.stdStepsPerCell_SL0, self.minStepsPerCell_SL0, self.maxStepsPerCell_SL0 = None, None, None, None
        self.meanStepsPerCell_SL1_SL8, self.stdStepsPerCell_SL1_SL8, self.minStepsPerCell_SL1_SL8, self.maxStepsPerCell_SL1_SL8 = None, None, None, None
        
        self.list_of_points = [] 
        rho_Vtx = np.sqrt( self.prodVtxX**2 +  self.prodVtxY**2 )
        for step in self.steps:
            p = Point(dictionary, instanceCDC, step)
            if p.inCDC:
                if rho_Vtx==0:
                    # save all the points from the POCA
                    self.list_of_points.append(p)
                else:
                    # save points from the production vertex of the particle
                    if p.rho>=rho_Vtx and rho_Vtx>0:
                        if self.pz>=0 and p.z>=self.prodVtxZ:
                            self.list_of_points.append(p)
                        if self.pz<0 and p.z<=self.prodVtxZ:
                            self.list_of_points.append(p)
            else:
                break
    
        dct = {'phi': [], 's': [], 'x':[], 'y':[], 'z': [], 'layer_id': [], 'cell_id':[]}
        for i in range(0, len(self.list_of_points)):
            dct['phi'].append(self.list_of_points[i].phi)
            dct['s'].append(self.list_of_points[i].s)
            dct['x'].append(self.list_of_points[i].x)
            dct['y'].append(self.list_of_points[i].y)
            dct['z'].append(self.list_of_points[i].z)
            dct['layer_id'].append(self.list_of_points[i].layer_id)
            dct['cell_id'].append(self.list_of_points[i].cell_id)
        self.points = pd.DataFrame(dct)
        # get number of steps in one cell
        # https://stackoverflow.com/questions/59946601/groupby-consecutive-occurrences-of-two-column-values-in-pandas
        self.points['cumSum'] = self.points[["layer_id","cell_id"]].ne(self.points[["layer_id","cell_id"]].shift()).any(axis=1).cumsum() 
        self.points['nStepsIncell']=self.points.groupby('cumSum')['layer_id'].transform('count')
        
        self._count_cdc_hits()
        
    def _count_cdc_hits(self):
        mask = (self.points.cell_id > -1)
        df_valid = self.points[mask]
        # as is, drop_duplicates will also drop cells if the track 'came back to the same cell after a time'
        # could use nStepsIncell to help with that or write something that drops only consecutive duplicates
        cells = df_valid.drop_duplicates(['layer_id', 'cell_id'], keep='first')
        self.CDChits = cells.loc[( ((cells.layer_id<8)&(cells.nStepsIncell>self.min_steps_in_SL0_cell)) | ((cells.layer_id>7)&(cells.nStepsIncell>self.min_steps_in_outer_cell)) ) ].shape[0]
    
        self.meanStepsPerCell = df_valid['nStepsIncell'].mean()
        self.stdStepsPerCell  = df_valid['nStepsIncell'].std()
        self.minStepsPerCell  = df_valid['nStepsIncell'].min()
        self.maxStepsPerCell  = df_valid['nStepsIncell'].max()

        self.meanStepsPerCell_SL0 = df_valid.query('layer_id<8')['nStepsIncell'].mean()
        self.stdStepsPerCell_SL0  = df_valid.query('layer_id<8')['nStepsIncell'].std()
        self.minStepsPerCell_SL0  = df_valid.query('layer_id<8')['nStepsIncell'].min()
        self.maxStepsPerCell_SL0  = df_valid.query('layer_id<8')['nStepsIncell'].max()

        self.meanStepsPerCell_SL1_SL8 = df_valid.query('layer_id>=8')['nStepsIncell'].mean()
        self.stdStepsPerCell_SL1_SL8  = df_valid.query('layer_id>=8')['nStepsIncell'].std()
        self.minStepsPerCell_SL1_SL8  = df_valid.query('layer_id>=8')['nStepsIncell'].min()
        self.maxStepsPerCell_SL1_SL8  = df_valid.query('layer_id>=8')['nStepsIncell'].max()

    # plots
    def xy_trajectory(self, ax, label='', color='b'):
        ax.scatter(self.points['x'], self.points['y'], color=color, label=label)

    def zy_trajectory(self, ax, label='', color='b'):
        ax.scatter(self.points['z'], self.points['y'], color=color, label=label)

    def xy_vertex(self, ax, label='', color='b'):
        ax.plot( [self.prodVtxX ], [self.prodVtxY], marker="*", markersize=15, color=color, label=label)
        
    def zy_vertex(self, ax, label='', color='b'):
        ax.plot( [self.prodVtxZ ], [self.prodVtxY], marker="*", markersize=15, color=color, label=label)
        
    def xy_poca(self, ax, label='', color='b'):
        ax.plot( [self.x0 ], [self.y0], marker="x", markersize=15, color=color, label=label)
        
    def zy_poca(self, ax, label='', color='b'):
        ax.plot( [self.z0 ], [self.y0], marker="x", markersize=15, color=color, label=label)
        