import numpy as np
import PIL
import sklearn.cluster as cluster

def _compute_player_mask(feature_screen, player_id):
    player_relative = np.array(feature_screen['player_relative'])
    mask = np.where(player_relative == player_id)
    return mask


def _compute_com(feature_screen, player_id):
    mask = _compute_player_mask(feature_screen, player_id)
    unit_density = np.array(feature_screen['unit_density'])
    rsum = []
    csum = []
    for r,c in zip(mask[0], mask[1]): 
        rsum.append(r * unit_density[r,c])
        csum.append(c *unit_density[r,c])
    if len(rsum) == 0:
        return None
    return int(round(np.mean(rsum))), int(round(np.mean(csum)))  # Mean row, mean col    

class BBox:
    def __init__(self, rl, rh, cl, ch):
        self.rl, self.rh, self.cl, self.ch = rl, rh, cl, ch
        self.width = ch - cl
        self.height = rh - rl
        self.ratio = self.height / max(self.width, 1)

def _compute_bbox(feature_screen, player_id):
    """ Computes the bounding box of all the players units"""
    mask = _compute_player_mask(feature_screen, player_id)
    row_low, row_high = np.min(mask[0]), np.max(mask[0])
    col_low, col_high = np.min(mask[1]), np.max(mask[1])
    return BBox(row_low, row_high, col_low, col_high)


def compute_density(feature_screen, player_id):
    mask = _compute_player_mask(feature_screen, player_id)
    unit_density = np.array(feature_screen['unit_density'])
    unit_mass = np.sum(unit_density[mask])
    bbox = _compute_bbox(feature_screen, player_id)
    overall_density = unit_mass / (bbox.width * bbox.height)
    return overall_density    


def compute_angle(v1, v2):
    if v2 is None:
        v2 = np.array([0, 1]) # Face east
    x = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_degrees = np.degrees(np.arccos(x))
    return cos_degrees

def compute_perpendicular(a, b):
    """ Computes perpendicular of a against b (from to a)"""
    a1 = np.dot(a, b) / np.linalg.norm(b)
    a2 = a - a1 # Facing vector
    return a2 

class Formation:
    def __init__(self, feature_screen, player_id, target_id):
        
        # Compute the center, left, and right flanks using K-Means
        mask = _compute_player_mask(feature_screen, player_id)
        as_coords = np.stack(mask).transpose()
        kmeans = cluster.KMeans(n_clusters=3, random_state=1337).fit(as_coords)
        cluster_assignments = kmeans.predict(as_coords)
        pts = []
        for idx in range(3):
            pts.append(np.round(np.mean(as_coords[np.where(cluster_assignments == idx)], axis=0)))
        # Populate distance matrix and assign the center point to be the cluster point
        # The other two points will be the outer points
        D = np.zeros((3,3))
        for src_idx in range(3):
            for tgt_idx in range(src_idx+1, 3):
                vec = pts[src_idx] - pts[tgt_idx]
                dist = np.linalg.norm(vec)
                D[src_idx, tgt_idx] = dist
                D[tgt_idx, src_idx] = dist
        summed_D = np.sum(D, axis=1)
        idx = np.where(summed_D == np.min(summed_D))[0][0]
        
        self.center_pt = pts[idx]
        self.outer_pts = []
        for i in range(3):
            if idx != i:
                self.outer_pts.append(pts[i])
                

        # Compute center of mass and bounding boxes
        self.own_com = _compute_com(feature_screen, player_id)
        self.target_com = _compute_com(feature_screen, target_id)
        self.bbox = _compute_bbox(feature_screen, player_id)
        self.box_density = compute_density(feature_screen, player_id)
                
        self.formation_angle = compute_angle(self.outer_pts[0] - self.center_pt, self.outer_pts[1] - self.center_pt)
        self.facing_vec = compute_perpendicular(self.center_pt - self.outer_pts[0], self.outer_pts[1] - self.outer_pts[0])
        self.facing_angle = compute_angle(self.facing_vec, None) # Angle with 0 at East
        if self.target_com is not None:
            self.target_vec = self.target_com - self.center_pt
            self.target_angle = compute_angle(self.target_vec, None)
            self.facing2target_angle = compute_angle(self.facing_vec, self.target_vec)
        self.formation = "n/a"
        
        # Determine self "triangle" facing versus target facing.
        # TODO: Determine ball
        if 0.9 <= abs(self.bbox.ratio) <= 1.3 and self.box_density >= 0.75:
            self.formation="ball"
        elif self.target_com is not None:
            if self.formation_angle >= 151:
                # File or column
                if self.facing2target_angle <= 30:
                    self.formation = "column" # line towards enemy
                elif 30 < self.facing2target_angle <60:
                    self.formation = "echelon"
                else:
                    self.formation = "file" # Line aimed at enemy
            else:
                # Concave or convex
                if 0 < self.facing2target_angle  < 75:
                    # Self triangle 'point' is facing target and we are curved, convex
                    self.formation = "convex"
                elif 105 <= self.facing2target_angle <= 180:
                    self.formation = "concave"
        else:
            self.formation = "None"
    
    def __str__(self):
        ret = [ "Center={}".format(self.center_pt),
                "Outer pts={}".format(self.outer_pts),
                "Formation Angle={:.3f}".format(self.formation_angle),
                "Perp vec={}".format(self.facing_vec),
                "Formation Facing Angle={:.3f}".format(self.facing_angle),
                "Formation={}".format(self.formation),
                "COM={}".format(self.own_com)]
        if self.target_vec is not None:
            ret.extend( [
                "Target COM={}".format(self.target_com), 
                "To Target Vec={}".format(self.target_vec),
                "Target Angle={:.3f}".format(self.target_angle),
                "Facing2Target Angle={:.3f}".format(self.facing2target_angle)
            ])
        return "\n".join(ret)
    
    def render(self, height=64, width=64, resized_width=256):
        X = PIL.Image.new('RGB', (height, width), (0,0,0))
        draw = PIL.ImageDraw.Draw(X)
        
        # Draw line between outer points
        draw.line((self.outer_pts[0][0], self.outer_pts[0][1], 
                   self.outer_pts[1][0], self.outer_pts[1][1]), fill='yellow')

        # Draw implied angle
        for outer_pt in self.outer_pts:
            draw.line((int(outer_pt[0]), int(outer_pt[1]), 
                       int(self.center_pt[0]), int(self.center_pt[1])), fill='blue')
            
        # Draw own center of mass
        draw.point((int(self.own_com[0]), int(self.own_com[1])), fill='green')
        
        # Draw own formatoin line center
        draw.point((int(self.center_pt[0]),
                    int(self.center_pt[1])), 
                   fill='yellow')

        if self.target_com is not None:
            # Draw enemy center of mass
            draw.point((int(self.target_com[0]), int(self.target_com[1])), fill='red')
            
        return PIL.ImageOps.flip(X.rotate(90)).resize((resized_width, resized_width))

    