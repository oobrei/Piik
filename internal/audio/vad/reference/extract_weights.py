import struct
data = open('/work/refbuild/dump.bin', 'rb').read()
print('total bytes', len(data))
sizes = [1008, 24, 1728, 1728, 72, 24, 1]
names = ['input_dense_weights', 'input_dense_bias', 'vad_gru_weights',
         'vad_gru_recurrent_weights', 'vad_gru_bias', 'vad_output_weights',
         'vad_output_bias']
off = 0
for n, s in zip(names, sizes):
    vals = struct.unpack('<%dh' % s, data[off:off + s * 2])
    print('%-28s %5d  first5 %s' % (n, s, list(vals[:5])))
    off += s * 2
