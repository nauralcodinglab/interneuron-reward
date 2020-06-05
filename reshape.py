from pathlib import Path

import h5py

import raw_io as rio

RAW_DATA_PATH = Path().joinpath('data', 'raw')
PROCESSED_DATA_PATH = Path().joinpath('data', 'processed', 'trials_by_celltype')

for sess in rio.walk_sessions(RAW_DATA_PATH):
    print('Saving {} {} {}'.format(sess.cell_type, sess.day, sess.mouse_id))
    try:
        with h5py.File(PROCESSED_DATA_PATH.joinpath(str(sess.day) + '.h5'), 'a') as f:
            f.require_group(sess.cell_type.as_path())
            f[sess.cell_type.as_path()].create_group(sess.mouse_id)
            sess.save(f[sess.cell_type.as_path()][sess.mouse_id])
            f.close()
    except rio.DataCorruptionError as der:
        print(der)
    except:
        print('Failed while saving {} {} {}'.format(sess.cell_type, sess.day, sess.mouse_id))
        raise

