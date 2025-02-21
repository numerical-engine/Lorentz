import numpy as np
from copy import deepcopy
import sys

class Model:
    """Lorentz96モデルのシミュレータ

    Attributes:
        x (np.ndarray): 現時刻の状態ベクトル。Shapeは(3, )
        p (float): パラメータ
        r (float): パラメータ
        b (float): パラメータ
    """
    def __init__(self, x_init:np.ndarray, p:float = 10., r:float = 28., b:float = 8./3.)->None:
        assert x_init.shape == (3, )
        self.x = x_init
        self.p = p
        self.r = r
        self.b = b
    
    def __call__(self)->np.ndarray:
        """現時刻の状態ベクトルを返す

        出力されるnp.ndarrayはdeepcopyされたもの

        Returns:
            np.ndarray: self.x
        """
        return deepcopy(self.x)
    
    def step(self, dt:float = 0.01)->None:
        """状態ベクトルを更新

        Args:
            dt (float, optional): 時間刻み。デフォルト値は0.01
        """
        x_b = deepcopy(self.x)

        self.x[0] = x_b[0] + dt*(-self.p*x_b[0] + self.p*x_b[1])
        self.x[1] = x_b[1] + dt*(-x_b[0]*x_b[2] + self.r*x_b[0] - x_b[1])
        self.x[2] = x_b[2] + dt*(x_b[0]*x_b[1] - self.b*x_b[2])