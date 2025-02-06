import pandas as pd
import math
import numpy as np

class PolygonFinder:
    def __init__(self, dataframe: pd.DataFrame, x='x', y='y', labels='labels'):
        self.df = dataframe
        self.x = dataframe[x].to_list()
        self.y = dataframe[y].to_list()
        self.labels = dataframe[labels].to_list()
    
    def get_point(self, id):
        return (self.x[id], self.y[id], id, self.labels[id])
    
    def get_area_points(self, selection_id, src_x, src_y, blacklist):
        result = []
        for i in selection_id:
            if i in blacklist:
                continue
            if self.x[i] > max(src_x) or self.x[i] < min(src_x):
                continue
            if self.y[i] > max(src_y) or self.y[i] < min(src_y):
                continue
            result.append(i)
        return result
    
    def get_area_inner_limits(self, selection_id, anchor):
        sel_x = [self.x[i] for i in selection_id]
        sel_y = [self.y[i] for i in selection_id]
        result = {}
        for a in anchor:
            if a == 'l':
                id = selection_id[np.argmin(sel_x)]
            elif a == 'r':
                id = selection_id[np.argmax(sel_x)]
            elif a == 'b':
                id = selection_id[np.argmin(sel_y)]
            elif a == 't':
                id = selection_id[np.argmax(sel_y)]
            point = self.get_point(id)
            result[a]= point
        return result
    
    def sort_anchor(self, anchor):
        anchor_order = 'ltrb'
        anchor = [c for c in anchor_order if c in anchor]
        if len(anchor) == 4:
            return 'ltrbl' # to make full round
        start_point = 0
        for i, a in enumerate(anchor_order):
            if a not in anchor:
                start_point = i + 1
        return (anchor_order + anchor_order)[start_point:start_point+len(anchor)]

    def limit_selection(self, selection_id, a, b):
        src_x = [a[0], b[0]]
        src_y = [a[1], b[1]]
        return self.get_area_points(selection_id, src_x, src_y, [a[2], b[2]])

    def get_borders(self, selection_id=None, anchor='ltrb', a=None, b=None, ungreedy=False):
        if selection_id is None:
            selection_id = range(len(self.x))
        if len(anchor) < 2:
            raise ValueError('Anchor must have 2 or more direction from list "ltrb"')
        if all([a is not None, b is not None]):
            if a and a[2] == b[2]:
                return [a]
            selection_id = self.limit_selection(selection_id, a, b)
            if not selection_id:
                return [a, b]
        anchor = self.sort_anchor(anchor)
        inner_limits = self.get_area_inner_limits(selection_id, anchor)
        pairs = [[anchor[i], anchor[i + 1]] for i in range(len(anchor) - 1)]
        result = [a, b] if a is not None else []
        for p in pairs:
            aa, bb = inner_limits[p[0]], inner_limits[p[1]]
            if a and ungreedy:
                filtered = self.ungreedy_filter(a, b, aa, bb, ''.join(p))
                if not filtered:
                    return result
                if not all(filtered):
                    return result + [p for p in filtered if p is not None]
            result += self.get_borders(selection_id, ''.join(p), aa, bb, ungreedy)
        return result
    
    def sin_points_to_x(self, a, b):
        """
        Get min sine for line ab to axis x
        """
        x = abs(a[0] - b[0])
        y = abs(a[1] - b[1])
        h = math.sqrt(x**2 + y**2)
        return y/h

    def sin_points_to_y(self, a, b):
        """
        Get min sine for line ab to axis x
        """
        x = abs(a[0] - b[0])
        y = abs(a[1] - b[1])
        h = math.sqrt(x**2 + y**2)
        return x/h
    
    def ungreedy_filter(self, a, b, aa, bb, anchor = 'tr'):
        sin_frame = self.sin_points_to_x(a, b)
        # base point is left or right
        base_point = b if (a[0] < b[0]) ^ ('l' in anchor) else a
        sin_aa = self.sin_points_to_x(base_point, aa)
        aa_result = (sin_aa > sin_frame) #Keep if outer point
        sin_bb = self.sin_points_to_x(base_point, bb)
        bb_result = (sin_bb > sin_frame) #Keep if outer point
        if all([aa_result, bb_result]):
        ## Check if need to keep both
        ## alt_point is top or bottom
            alt_point = b if (a[1] < b[1]) ^ ('b' in anchor) else a
            alt_sin_aa = self.sin_points_to_x(alt_point, aa)
            alt_sin_bb = self.sin_points_to_x(alt_point, bb)
            aa_result = not (sin_aa < sin_bb) & (alt_sin_aa > alt_sin_bb)            
            bb_result = not (sin_bb < sin_aa) & (alt_sin_bb > alt_sin_aa)
            if not [aa_result, bb_result]: #Keep both if they have same evaluation
                aa_result = bb_result = True
        return aa if aa_result else None, bb if bb_result else None