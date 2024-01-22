"""
Create pretty moving-dot animations using matplotlib based on
the rotation of a hypercube vertices
"""

from dataclasses import dataclass
import math
from random import shuffle, choice, randrange
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
   

def int_to_binary(val: int, n: Optional[int] = None) -> List[int]:
    if n is None:
        return [int(char) for char in f'{val:b}']
    return [int(char) for char in f'{val:0{n}b}']


def binary_to_int(bit_list: List[int]) -> int:
    return int(''.join(str(x) for x in bit_list), 2)


@dataclass
class Pos:
    """2D co-ordinate"""
    
    x: float
    y: float
    
    def __add__(self, other: 'Pos') -> 'Pos':
        assert isinstance(other, Pos)
        return Pos(self.x + other.x, self.y + other.y)
    
    def __radd__(self, other: 'Pos') -> 'Pos':
        assert isinstance(other, Pos)
        return Pos(self.x + other.x, self.y + other.y)
    
    def __rmul__(self, other: Union[int, float]) -> 'Pos':
        assert isinstance(other, (int, float))
        return Pos(other * self.x, other * self.y)
    
    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Dot:
    """Single animated dot"""
    
    val: int  # this identifies vertex of hypercube when converted to binary
    colour: str
    pos_list: List[Pos]  # defines path the dot follows
    
    def get_pos(self, time: float) -> Tuple[float, float]:
        pos_from = self.pos_list[int(time) % len(self.pos_list)]
        pos_to = self.pos_list[int(time + 1) % len(self.pos_list)]
        #time = max(0, min(1, (time % 1) * 1.1 - 0.05))
        ratio = math.sin((time % 1) * math.pi/2)**3
        return (1-ratio) * pos_from + ratio * pos_to
    

@dataclass
class Dots:
    """Container for all dots"""
    
    dot_list: List[Dot]
    
    def offset_list(self, t: float) -> List[Tuple[float, float]]:
        return [dot.get_pos(t).as_tuple() for dot in self.dot_list]
    
    def x0_list(self) -> List[float]:
        return [dot.pos_list[0].x for dot in self.dot_list]
    
    def y0_list(self) -> List[float]:
        return [dot.pos_list[0].y for dot in self.dot_list]
    
    def colour_array(self) -> List[str]:
        return [dot.colour for dot in self.dot_list]
    

    
@dataclass
class Transform:
    """Determines rotation/reflection of hypercube"""
    
    n: int
    reorderings: Optional[List[int]] = None
    reflections: Optional[List[bool]] = None
    
    @staticmethod
    def random(n: int) -> 'Transform':
        reorderings: Dict[int, int] = {}
        reorderings = list(range(n))
        shuffle(reorderings)
        reflections = [choice([True, False]) for _ in range(n)]
        return Transform(n, reorderings, reflections)
    
    def from_str(s: str) -> 'Transform':
        n = len(s.split(','))
        reorderings = [abs(int(x)) for x in s.split(',')]
        reflections = [x.startswith('-') for x in s.split(',')]
        return Transform(n, reorderings, reflections)
    
    def apply(self, val: int) -> int:
        bit_list = int_to_binary(val, n=self.n)
        if self.reorderings is not None:
            bit_list = [bit_list[i] for i in self.reorderings]
        if self.reflections is not None:
            bit_list = [1-bit_list[i] if reflect else bit_list[i] for i, reflect in enumerate(self.reflections)]
        return binary_to_int(bit_list)
    
    def __repr__(self) -> str:
        s = ','.join((f'-{x}' if reflect else f'{x}' for x, reflect in zip(self.reorderings, self.reflections)))
        return f'Transform.from_str("{s}")'

    
@dataclass
class Layout:
    """Defines where in 2D-space a hypercube vertex maps to"""
    
    n: int
    offset: Pos
    basis_vectors: List[Pos]
    
    @staticmethod
    def standard(n: int) -> 'Layout':
        grouping_factor = 1.0
        basis_vectors = []
        offset = Pos(1, 1)
        for i in range(n):
            length = (grouping_factor/2)**(1+i//2)
            if i % 2 == 0:
                basis_vectors.append(Pos(length, 0))
            else:
                basis_vectors.append(Pos(0, length))
            offset += -1 * basis_vectors[-1]
        offset = 0.5 * offset
        return Layout(n, offset, basis_vectors)
    
    def calc_position(self, val: int) -> Pos:
        bit_list = int_to_binary(val, n=self.n)
        pos = self.offset
        for bit, vect in zip(bit_list, self.basis_vectors):
            pos += bit*vect
        return pos
    
    
@dataclass
class Colour:
    """RGB colour"""
    
    r: int
    g: int
    b: int
    
    @staticmethod
    def random() -> 'Colour':
        return Colour(randrange(256), randrange(256), randrange(256))
    
    def random_delta() -> 'Colour':
        return Colour(randrange(256) - 128, randrange(256) - 128, randrange(256) - 128)
    
    def to_hex(self) -> str:
        return "#{0:02x}{1:02x}{2:02x}".format(
            max(0, min(255, self.r)),
            max(0, min(255, self.g)),
            max(0, min(255, self.b)),
        )
    
    def __add__(self, other: 'Colour') -> 'Colour':
        return Colour(self.r + other.r, self.g + other.g, self.b + other.b)
    
    def __sub__(self, other: 'Colour') -> 'Colour':
        return Colour(self.r - other.r, self.g - other.g, self.b - other.b)
    
    def __mul__(self, other: Union[int, float]) -> 'Colour':
        return Colour(int(other*self.r), int(other*self.g), int(other*self.b))
    
    def __rmul__(self, other: Union[int, float]) -> 'Colour':
        return Colour(int(other*self.r), int(other*self.g), int(other*self.b))
    
    def __repr__(self) -> str:
        return f'Colour({self.r},{self.g},{self.b})'
    

@dataclass
class Palette:
    """Picks a random 2D colour gradient based on positions in 2D space"""

    origin_colour: Colour = Colour.random()
    x_colour_delta: Colour = Colour.random_delta()
    y_colour_delta: Colour = Colour.random_delta()
    
    def calc_colour(self, layout: Layout, val: int) -> str:
        pos = layout.calc_position(val)
        colour = self.origin_colour + pos.x*self.x_colour_delta + pos.y*self.y_colour_delta
        return colour.to_hex()
    
    def __repr__(self) -> str:
        return (
            f'Palette({repr(self.origin_colour)}, '
            f'{repr(self.x_colour_delta)}, {repr(self.y_colour_delta)})'
        )
        
    
@dataclass
class Animator:
    """Creates dots based on selected transform, layout and colour palette"""
    
    n: int
    transform: Transform
    layout: Layout
    palette: Palette
    latest_animation: Optional[FuncAnimation] = None
    
    def animate(self) -> FuncAnimation:
        dot_list: List[Dot] = []
        for val0 in range(2**self.n):
            val = val0
            pos_list: List[Pos] = []
            while True:
                pos_list.append(self.layout.calc_position(val))
                val = self.transform.apply(val)
                if val == val0:
                    break
            colour = self.palette.calc_colour(self.layout, val0)
            dot_list.append(Dot(val0, colour, pos_list))
        dots = Dots(dot_list)
        
        fig = plt.figure(figsize=(7, 7))
        fig.patch.set_facecolor('#202020')
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.set_xlim(0, 1), ax.set_xticks([])
        ax.set_ylim(0, 1), ax.set_yticks([])
        scat = ax.scatter(
            dots.x0_list(),
            dots.y0_list(),
            facecolors=dots.colour_array(),
            s=700,
            lw=0,
            alpha=0.6,
        )
        
        def update(frame_number):
            time = frame_number / 200
            scat.set_offsets(dots.offset_list(time))
            
        self.latest_animation = FuncAnimation(
            fig, update, interval=20, save_count=100
        )
        plt.show()        
    
    def __repr__(self) -> str:
        return (
            f'DotsCreator(n={n}, transform={repr(self.transform)}, '
            f'layout=Layout.standard({n}), palette={repr(self.palette)})'
        )


n = 8  # number of dimensions of hypercube
transform = Transform.random(n)
# transform = Transform.from_str('-1,0,3,-2,4,5,7,-6')
# transform = Transform.from_str('7,6,5,4,3,2,1,0')
# transform = Transform.from_str('2,3,4,5,6,7,0,1')
layout = Layout.standard(n)
palette = Palette()
animator = Animator(n, transform, layout, palette)
print(repr(animator) + '.create().animate()')
animator.animate()
