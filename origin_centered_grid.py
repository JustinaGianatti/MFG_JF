#! python3

class ZGrid(object):
    ''' Helper class to represent a squared grid center on the origin.
    Given the following centered grid of with (2N+1)^2 points:

                  ^
                  |
          a * * * | * * * *
          * * * * | * * * *
          * * * * | * * * *
          * * * * | * * * *
       <----------0----------->
          * * * * | * * * *
          * * * * | * * * *
          * * * * | * * b *
          * * * * | * * * *
                  |

    We want to be able to access elemen `a` as grid[-N,N] and element `b` as grid[N-1, -(N-1)].

    Using this grid, we should be able to access and asign values using the "natural" indexing.
        The grid is stored as a single list following this rules:
            - The first 2n+1 elements correspond to the points of the form (x,N) for x in [-N,N].
            - The second 2n+1 elements correspond to the points of the form (x,N-1) for x in [-N, N].
              ...'''


    def __init__(self, radius, init_value = 0):
        ''' Initializes a grid of radius `radius` center in the origin with values `init_value` for
        all points.'''
        self._radius = radius
        self._grilla = [init_value] * ((2 * radius + 1) ** 2)
        self._max_idx = (2 * radius + 1) ** 2 - 1

    @property
    def radius(self):
        return self._radius

    def _x_y_to_index(self, x, y):
        radius = self.radius
        if abs(x) > radius or abs(y) > radius:
            raise IndexError('({},{}) is out of the grid. (Grid radius {})'.format(x, y, radius))
        idx = x + radius + (radius - y) * (2 * radius + 1)
        if not (0 <= idx <= self._max_idx):
            raise IndexError('({},{}) is out of the grid. (Grid radius/Computed idx/Max idx: {}/{}/{})'.format(x, y, radius, idx, self._max_idx))
        return idx

    def __getitem__(self, pair):
        if not isinstance(pair, tuple) or (len(pair) != 2) or not isinstance(pair[0], int) or not isinstance(pair[1], int):
            raise TypeError('Grid indices must be a two integral coordinates x,y, not ', type(pair))
        return self._grilla[self._x_y_to_index(*pair)]

    def __setitem__(self, pair, value):
        if not isinstance(pair, tuple) or (len(pair) != 2) or not isinstance(pair[0], int) or not isinstance(pair[1], int):
            raise TypeError('Grid indices must be a two integral coordinates x,y, not ', type(pair))
        self._grilla[self._x_y_to_index(*pair)] = value

    def __repr__(self):
        return super().__repr__()

    def pretty_str(self):
        grid = []
        for y in range(self.radius, -self.radius - 1, -1):
            row = []
            for x in range(-self.radius, self.radius + 1):
                if x == y and x == 0:
                    row.append('Z')
                else:
                    row.append(str(self[x,y]))
            grid.append('\t'.join(row))
        grid_str = '\n'.join(grid)
        return "Origin centered grid of radius {}\n{}".format(self.radius, grid_str)



if __name__ == '__main__':
    g1 = ZGrid(1)
    assert g1.radius == 1
    print(g1.pretty_str())

    g4 = ZGrid(4)
    assert g4.radius == 4
    assert g4[3,3] == 0
    g4[3,3] = 42
    assert g4[3,3] == 42
    print(g4.pretty_str())

    r = 3
    g3 = ZGrid(r)
    assert g3.radius == r
    for x in range(-g3.radius, g3.radius + 1):
        for y in range(-g3.radius, g3.radius + 1):
            g3[x, y] = g3._x_y_to_index(x, y)
    print(g3.pretty_str())
