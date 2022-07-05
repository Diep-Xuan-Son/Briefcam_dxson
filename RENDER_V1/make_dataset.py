import torch
import numpy as np
from io import BytesIO
from PIL import Image
import struct, cv2
from scipy.io import loadmat
from collections import defaultdict
def read_vbb(path):
    assert path[-3:] == 'vbb'
    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                # MATLAB is 1-origin
                datum['str'] = int(objStr[datum['id']]) - 1
                # MATLAB is 1-origin
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data
def read_seq(path):
    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert(length != 1024)
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(0,9)]
        fps = struct.unpack('@d', ifile.read(8))
        # skipping the rest
        ifile.read(432)
        image_ext = {100:'raw', 102:'jpg',201:'jpg',1:'png',2:'png'}
        return {'w':params[0],'h':params[1],
                'bdepth':params[2],
                'ext':image_ext[params[5]],
                'format':params[5],
                'size':params[4],
                'true_size':params[8],
                'num_frames':params[6]}
    
    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    # this is freaking magic, but it works
    extra = 8
    s = 1024
    seek = [0]*(params['num_frames']+1)
    seek[0] = 1024
    
    images = []
    
    for i in range(0, params['num_frames']-1):
        tmp = struct.unpack_from('@I', bytes[s:s+4])[0]
        s = seek[i] + tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s+1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        seek[i+1] = s
        nbytes = struct.unpack_from('@i', bytes[s:s+4])[0]
        I = bytes[s+4:s+nbytes]
        
        tmp_file = 'H:\\Object_detection\\person_vehicle\\USA\\V000\\%d.jpg' % i
        open(tmp_file, 'wb+').write(I)
        
        img = cv2.imread(tmp_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images
# img = read_seq("H:\\Object_detection\\person_vehicle\\USA\\V000.seq")
# for i in img:
#     cv2.imshow("1", i)
#     cv2.waitKey(1)
data = read_vbb("H:\\Object_detection\\person_vehicle\\USA\\V000.vbb")
print(data)