# coding: utf-8
from __future__ import print_function, division, absolute_import
import os
import re
import copy
import numpy as np

# NOTE array indices are converted to be zero-based

class TetgenNodes:
    def __init__(self):
        self.num_points = 0
        self.dim = 0
        self.points = None
        self.num_attrs = 0
        self.attrs = None
        self.has_boundary_markers = 0
        self.boundary_markers = None

    def load(self, filename):
        # (1) read .1.node file (points)
        with open(filename) as f:
            # first line: <# of points> <dimension> <# of attributes> <boundary markers (0 or 1)>
            num_points, dim, num_attrs, has_boundary_markers = [int(x) for x in re.split(r'\s+',f.readline().strip())]
            points = np.zeros([num_points,dim],dtype=float)
            attrs  = np.zeros([num_points,num_attrs],dtype=float) if num_attrs > 0 else np.empty(0)
            boundary_markers = np.zeros([num_points],dtype=int) if has_boundary_markers > 0 else np.empty(0)
            for i in range(num_points):
                # <point #> <x> <y> <z> [attributes] [boundary marker]
                items = re.split(r'\s+',f.readline().strip())
                assert int(items[0]) == i + 1, ('items[0]',items[0],'i + 1',i + 1)
                points[i,:dim] = [float(x) for x in items[1:1+dim]]
                if num_attrs > 0:
                    attrs[i,:num_attrs] = [float(x) for x in items[1+dim:1+dim+num_attrs]]
                if has_boundary_markers > 0:
                    boundary_markers[i] = int(items[-1])

        self.num_points = num_points
        self.dim = dim
        self.num_attrs = num_attrs
        self.has_boundary_markers = has_boundary_markers
        self.points = points
        self.attrs = attrs
        self.boundary_markers = boundary_markers

    def save(self, filename):
        with open(filename,'w') as f:
            assert len(self.points) == self.num_points
            # first line: <# of points> <dimension> <# of attributes> <boundary markers (0 or 1)>
            f.write('{:d} {:d} {:d} {:d}\n'.format(
                len(self.points),
                self.dim,
                self.num_attrs,
                self.has_boundary_markers))
            for i,p in enumerate(self.points):
                assert len(p) == self.dim
                # <point #> <x> <y> <z> [attributes] [boundary marker]
                line = '{:d}'.format(i+1)
                for _,v in enumerate(p):
                    line += ' {:f}'.format(v)
                if self.num_attrs:
                    assert len(self.attrs[i]) == self.num_attrs
                    for _,a in enumerate(self.attrs[i]):
                        line += ' {:f}'.format(a)
                if self.has_boundary_markers:
                    line += ' {:d}'.format(self.boundary_markers[i])
                f.write(line + '\n')


class TeggenElems:
    def __init__(self):
        self.num_elems = 0
        self.num_nodes = 0
        self.elems = None
        self.num_attrs = 0
        self.attrs = None

    def load(self,filename):
        # (3) read .1.ele file (ele == tetrahedron)
        with open(filename) as f:
            # first line: <# of tetrahedra> <# of nodes> <# of attributes>
            num_elems, num_nodes, num_attrs = [int(x) for x in re.split(r'\s+',f.readline().strip())]
            elems = np.zeros([num_elems, num_nodes],dtype=int)
            attrs = np.zeros([num_elems,num_attrs],dtype=int) if num_attrs > 0 else np.empty(0)
            for i in range(num_elems):
                # <ele #> <node> <node> <node> ... [attributes]
                items = re.split(r'\s+',f.readline().strip())
                assert int(items[0]) == i + 1, ('items[0]',items[0],'i + 1',i + 1)
                elems[i,:num_nodes] = [int(x) - 1 for x in items[1:1+num_nodes]]
                if num_attrs > 0:
                    attrs[i,:num_attrs] = [float(x) for x in items[1+num_nodes:1+num_nodes+num_attrs]]

        self.num_elems = num_elems
        self.num_nodes = num_nodes
        self.num_attrs = num_attrs
        self.elems = elems
        self.attrs = attrs

    def save(self,filename):
        with open(filename,'w') as f:
            assert len(self.elems) == self.num_elems
            # first line: <# of tetrahedra> <# of nodes> <# of attributes>
            f.write('{:d} {:d} {:d}\n'.format(
                self.num_elems,
                self.num_nodes,
                self.num_attrs))
            for i,e in enumerate(self.elems):
                assert len(e) == self.num_nodes
                # <ele #> <node> <node> <node> ... [attributes]
                line = '{:d}'.format(i+1)
                for _,v in enumerate(e):
                    line += ' {:d}'.format(v+1)
                if self.num_attrs:
                    assert len(self.attrs[i]) == self.num_attrs
                    for _,a in enumerate(self.attrs[i]):
                        line += ' {:f}'.format(a)
                f.write(line + '\n')


class TetgenFaces:
    def __init__(self):
        self.num_faces = 0
        self.faces = None
        self.has_boundary_markers = 0
        self.boundary_markers = None

    def load(self,filename):
        # (2) read .1.face file (faces == triangles)
        with open(filename) as f:
            # first line: <# of faces> <boundary markers (0 or 1)>
            num_faces, has_boundary_markers = [int(x) for x in re.split(r'\s+',f.readline().strip())]
            faces = np.zeros([num_faces,3],dtype=int)
            boundary_markers = np.zeros([num_faces],dtype=int) if has_boundary_markers > 0 else np.empty(0)
            for i in range(num_faces):
                # <face #> <node> <node> <node> [boundary marker]
                items = re.split(r'\s+',f.readline().strip())
                assert int(items[0]) == i + 1, ('items[0]',items[0],'i + 1',i + 1)
                faces[i,:3] = [int(x)-1 for x in items[1:4]]
                if has_boundary_markers > 0:
                    boundary_markers[i] = int(items[-1])

        self.num_faces = num_faces
        self.has_boundary_markers = has_boundary_markers
        self.faces = faces
        self.boundary_markers = boundary_markers

    def save(self,filename):
        with open(filename,'w') as f:
            assert len(self.faces) == self.num_faces
            # first line: <# of faces> <boundary markers (0 or 1)>
            f.write('{:d} {:d}\n'.format(
                self.num_faces,
                self.has_boundary_markers))
            for i,face in enumerate(self.faces):
                assert len(face) == 3
                # <face #> <node> <node> <node> [boundary marker]
                line = '{:d}'.format(i+1)
                for _,v in enumerate(face):
                    line += ' {:d}'.format(v+1)
                if self.has_boundary_markers:
                    line += ' {:d}'.format(self.boundary_markers[i])
                f.write(line + '\n')


class TetgenObject:
    def __init__(self):
        self.nodes = TetgenNodes()
        self.elems = TeggenElems()
        self.faces = TetgenFaces()

    def load(self, model_file):
        # automatic detect filename base
        rev = model_file[::-1]
        if rev[:5] == 'edon.' or rev[:5] == 'ecaf.': # .node or .face
            model_file_base = model_file[:-5]
        elif rev[:5] == 'ele.': # .ele
            model_file_base = model_file[:-4]
        else:
            model_file_base = model_file

        assert os.access(model_file_base + '.node',os.R_OK)
        assert os.access(model_file_base + '.ele',os.R_OK)
        assert os.access(model_file_base + '.face',os.R_OK)

        if os.access(model_file_base + '.node',os.R_OK):
            self.nodes.load(model_file_base + '.node')
        if os.access(model_file_base + '.ele',os.R_OK):
            self.elems.load(model_file_base + '.ele')
        if os.access(model_file_base + '.face',os.R_OK):
            self.faces.load(model_file_base + '.face')

    def save(self, model_file_base):
        self.nodes.save(model_file_base + '.node')
        self.elems.save(model_file_base + '.ele')
        self.faces.save(model_file_base + '.face')


    def rebuild(self, selected_elems, elem_attr=None):
        """
        create a new TetgenObject composed of selected_elems
        """
        new_points_map = dict()
        new_points_index = 0
        for elem in selected_elems:
            for n in elem:
                if not n in new_points_map:
                    new_points_map[n] = new_points_index
                    new_points_index += 1

        new_points_ref = np.zeros(len(new_points_map),dtype=int)
        for k,v in new_points_map.items():
            new_points_ref[v] = k

        new_points = np.zeros([len(new_points_ref),3],dtype=float)
        if self.nodes.num_attrs > 0:
            new_node_attrs = np.zeros([len(new_points_ref),self.nodes.num_attrs],dtype=float)
        if self.nodes.has_boundary_markers > 0:
            new_node_boundary_markers = np.zeros(len(new_points_ref),dtype=int)
        for i,pos in enumerate(new_points_ref):
            new_points[i] = self.nodes.points[pos]
            if self.nodes.num_attrs > 0:
                new_node_attrs[i,:] = self.nodes.attrs[pos,:]
            if self.nodes.has_boundary_markers > 0:
                new_node_boundary_markers[i] = self.nodes.boundary_markers[pos]

        new_elems = np.zeros_like(selected_elems)
        for i,elem in enumerate(selected_elems):
            a, b, c, d = elem
            new_elems[i] = new_points_map[a],new_points_map[b],new_points_map[c],new_points_map[d]

        new_faces = elems_to_faces(new_elems)

        obj2 = TetgenObject()

        obj2.elems.elems = new_elems
        if elem_attr is not None:
            obj2.elems.attrs = np.zeros_like(elem_attr)
            obj2.elems.attrs[:] = elem_attr
            obj2.elems.num_attrs = len(elem_attr[0])
        obj2.elems.num_nodes = 4
        obj2.elems.num_elems = len(new_elems)

        obj2.faces.faces = new_faces
        obj2.faces.num_faces = len(new_faces)

        obj2.nodes.points = new_points
        obj2.nodes.num_points = len(new_points)
        obj2.nodes.dim = 3
        if self.nodes.num_attrs > 0:
            obj2.nodes.attrs = new_node_attrs
            obj2.nodes.num_attrs = len(new_node_attrs[0])
        if self.nodes.has_boundary_markers > 0:
            obj2.nodes.boundary_markers = new_node_boundary_markers
            obj2.nodes.has_boundary_markers = 1

        return obj2



def elems_to_faces(elems,permute=False,keepdims=False,ccw=True):
    assert elems.ndim == 2
    assert elems.shape[1] == 4
    if ccw:
        e2f = [[0,2,1],[0,1,3],[1,2,3],[0,3,2]] # 1-3-2, 1-2-4, 2-3-4, 1-4-3 (~4-3-1)
    else:
        e2f = [[0,3,2],[1,3,2],[2,3,0],[0,1,2]] # 1-4-3 2-4-3 3-4-1 1-2-3
    t1 = elems[:,e2f[0]]
    t2 = elems[:,e2f[1]]
    t3 = elems[:,e2f[2]]
    t4 = elems[:,e2f[3]]
    t_ = np.transpose([t1,t2,t3,t4],[1,0,2])
    if permute:
        t2_ = t_[:,:,[1,2,0]] # rot 1
        t3_ = t_[:,:,[2,0,1]] # rot 2
        t_  = np.concatenate([t_,t2_,t3_],axis=0)
    if not keepdims:
        t_ = t_.reshape([-1,3])
    return t_

def load_tetgen(model_file):
    """
    load tetgen output ( .node & .face )
    """

    tetgen_obj = TetgenObject()
    tetgen_obj.load(model_file)
    return tetgen_obj

# vertex 의 group code 결정
def find_vertex_group(uvcoords, voronoi_points, voronoi_group):
    assert uvcoords.ndim == 2
    assert uvcoords.shape[1] == 2
    uvcoords = uvcoords.reshape([-1,1,2])
    offs = uvcoords - voronoi_points # shape (-1, N, 2)
    dists = np.linalg.norm(offs,axis=-1) # shape (-1, N)
    center = np.argmin(dists, axis=-1) # get nearest center, shape (-1)
    v_group = voronoi_group[center]
    return -(2+v_group) # -2 부터 시작해서 감소하도록 맵


# face (triangle) 혹은 element (tetra) 의 group 코드 결정
def find_element_group(elems, texcoords, voronoi_points, voronoi_group):
    assert elems.ndim == 2
    assert texcoords.ndim == 2
    assert texcoords.shape[1] == 2
    assert voronoi_points.ndim == 2
    assert voronoi_points.shape[1] == 2
    uvcoords = np.zeros([len(elems),1,2],dtype=float) # 모든 elements 들에 대해 u, v 값
    uvcoords[:,0,0] = np.amax(texcoords[elems][:,:,0],axis=1) # u 방향으로는 max 값 선택
    # uvcoords[:,0,1] = np.mean(texcoords[elems][:,:,1],axis=1) # v 방향으로는 mean 값 선택
    uvcoords[:,0,1] = np.amax(texcoords[elems][:,:,1],axis=1) # v 방향으로는 amax 값 선택
    offs = uvcoords - voronoi_points # shape (-1, N, 2)
    dists = np.linalg.norm(offs,axis=-1) # shape (-1, N)
    center = np.argmin(dists, axis=-1) # get nearest center, shape (-1)
    v_group = voronoi_group[center]
    return -(2+v_group) # -2 부터 시작해서 감소하도록 맵

def rebuild_submesh(obj1, selection, voronoi_points, voronoi_group):
    """
    create a new TetgenObject composed of selected_elems
    selection <= -2
    """

    assert obj1.nodes.num_attrs > 0
    assert obj1.nodes.has_boundary_markers > 0
    assert obj1.faces.has_boundary_markers > 0

    points           = obj1.nodes.points
    marks            = obj1.nodes.boundary_markers
    texcoords        = obj1.nodes.attrs
    elems            = obj1.elems.elems
    faces            = obj1.faces.faces
    face_group       = obj1.faces.boundary_markers

    elem_group       = find_element_group(elems,texcoords,voronoi_points,voronoi_group)

    elem_select    = elem_group == selection
    face_select    = face_group == selection

    # select relevant vertices
    vert_selected  = np.zeros(len(points),dtype=int)
    for elem in elems[elem_select]:
        a, b, c, d = elem
        vert_selected[[a,b,c,d]] = 1
    for face in faces[face_select]:
        a, b, c = face
        vert_selected[[a,b,c]] = 1
    vertex_select  = vert_selected == 1

    vertex_remap   = np.zeros(len(points),dtype=int)
    vertex_remap[:] = -(len(points)+1) # some invalid value
    for i,v in enumerate(np.arange(len(points),dtype=int)[vertex_select]):
        vertex_remap[v] = i

    obj2 = TetgenObject()

    obj2.nodes.dim  = obj1.nodes.dim
    obj2.nodes.num_attrs  = obj1.nodes.num_attrs
    obj2.nodes.has_boundary_markers = obj1.nodes.has_boundary_markers

    obj2.nodes.points = points[vertex_select]
    obj2.nodes.attrs  = texcoords[vertex_select]
    obj2.nodes.boundary_markers = marks[vertex_select]
    obj2.nodes.num_points = len(obj2.nodes.points)

    obj2.elems.num_attrs = obj1.elems.num_attrs
    obj2.elems.num_nodes = obj1.elems.num_nodes

    obj2.elems.elems     = vertex_remap[elems[elem_select]]
    obj2.elems.attrs     = np.empty(0)
    if obj1.elems.num_attrs > 0:
        obj2.elems.attrs = obj1.elems.attrs[elem_select]
    obj2.elems.num_elems = len(obj2.elems.elems)

    obj2.faces.has_boundary_markers = obj1.faces.has_boundary_markers

    obj2.faces.faces     = vertex_remap[faces[face_select]]
    obj2.faces.boundary_markers = face_group[face_select]
    obj2.faces.num_faces = len(obj2.faces.faces)

    return obj2


def elems_to_faces2(elems):
    assert elems.ndim == 2
    assert elems.shape[1] == 4
    e2f = [[0,2,1],[0,1,3],[1,2,3],[0,3,2]] # 1-3-2, 1-2-4, 2-3-4, 1-4-3 (~4-3-1)
    t1 = elems[:,e2f[0]]
    t2 = elems[:,e2f[1]]
    t3 = elems[:,e2f[2]]
    t4 = elems[:,e2f[3]]
    t_ = np.sort(np.transpose([t1,t2,t3,t4],[1,0,2]).reshape([-1,3]),axis=-1).tolist()
    # count and remove duplicates
    face_dict = dict()
    for f in t_:
        ft = tuple(f)
        if ft not in face_dict:
            face_dict[ft] = 1
        else:
            face_dict[ft] += 1
    faces = np.zeros([len(face_dict),3],dtype=int)
    counts = np.zeros(len(face_dict),dtype=int)
    for i,en in enumerate(face_dict.items()):
        f, c = en
        faces[i,:] = f
        counts[i] = c
    return faces, counts


def rebuild_submesh2(obj1, selection, voronoi_points, voronoi_group):
    """
    create a new TetgenObject composed of selected_elems
    selection <= -2
    """

    assert obj1.nodes.num_attrs > 0
    assert obj1.nodes.has_boundary_markers > 0

    points           = obj1.nodes.points
    marks            = obj1.nodes.boundary_markers
    texcoords        = obj1.nodes.attrs
    elems            = obj1.elems.elems

    elem_group       = find_element_group(elems,texcoords,voronoi_points,voronoi_group)
    elem_select      = elem_group == selection

    # select relevant vertices
    vert_selected  = np.zeros(len(points),dtype=int)
    for elem in elems[elem_select]:
        a, b, c, d = elem
        vert_selected[[a,b,c,d]] = 1
    vertex_select  = vert_selected == 1

    vertex_remap   = np.zeros(len(points),dtype=int)
    vertex_remap[:] = -(len(points)+1) # some invalid value
    for i,v in enumerate(np.arange(len(points),dtype=int)[vertex_select]):
        vertex_remap[v] = i

    obj2 = TetgenObject()

    obj2.nodes.dim  = obj1.nodes.dim
    obj2.nodes.num_attrs  = obj1.nodes.num_attrs
    obj2.nodes.has_boundary_markers = obj1.nodes.has_boundary_markers

    obj2.nodes.points = points[vertex_select]
    obj2.nodes.attrs  = texcoords[vertex_select]
    obj2.nodes.boundary_markers = marks[vertex_select]
    obj2.nodes.num_points = len(obj2.nodes.points)

    obj2.elems.num_attrs = obj1.elems.num_attrs
    obj2.elems.num_nodes = obj1.elems.num_nodes

    obj2.elems.elems     = vertex_remap[elems[elem_select]]
    obj2.elems.attrs     = np.empty(0)
    if obj1.elems.num_attrs > 0:
        obj2.elems.attrs = obj1.elems.attrs[elem_select]
    obj2.elems.num_elems = len(obj2.elems.elems)

    obj2.faces.has_boundary_markers = 1

    faces, counts        = elems_to_faces2(obj2.elems.elems)
    obj2.faces.faces     = faces
    obj2.faces.num_faces = len(obj2.faces.faces)
    obj2.faces.boundary_markers = np.zeros(obj2.faces.num_faces,dtype=int)
    obj2.faces.boundary_markers[:] = -1 # 중복된 face 는 (내부) -1
    obj2.faces.boundary_markers[counts == 1] = selection  # 중복되지 않은 face 는 원본 selection 값

    # create a point-cloud array
    ptcloud_faces = faces[counts == 1]
    ptcloud       = np.mean(obj2.nodes.points[ptcloud_faces],axis=1)

    return obj2, ptcloud

