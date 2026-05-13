"""Create a tiny (<5 MB) sample HDF5 from a real episode — just enough to
illustrate the format. We keep the first 16 frames, all attributes, and
all action/state streams, but truncate the 256x288 image streams hard."""
from __future__ import annotations
import h5py, numpy as np
from pathlib import Path

SRC = Path("/home/ubuntu/ws_aic/data/episodes_v2_balanced_0p5/episode_1778471633587_20a19af2.h5")
DST = Path("/home/ubuntu/ws_aic/report/data_sample/sample_episode.h5")
N = 16  # keep only 16 frames (~1 second at 16.7 Hz)


def copy_truncated(src, dst, n=N):
    if isinstance(src, h5py.Dataset):
        data = src[:n] if src.shape and src.shape[0] > n else src[...]
        d = dst.create_dataset(src.name.split("/")[-1], data=data,
                               compression="gzip", compression_opts=4)
        for k, v in src.attrs.items():
            d.attrs[k] = v
        return
    # group
    for k in src.keys():
        if isinstance(src[k], h5py.Group):
            g = dst.create_group(k)
            for ak, av in src[k].attrs.items():
                g.attrs[ak] = av
            copy_truncated(src[k], g, n)
        else:
            copy_truncated(src[k], dst, n)


with h5py.File(SRC, "r") as fin, h5py.File(DST, "w") as fout:
    for k, v in fin.attrs.items():
        fout.attrs[k] = v
    fout.attrs["__sample_note__"] = (
        f"Truncated to first {N} frames for repository inclusion. "
        "See REPORT.md for full dataset description."
    )
    copy_truncated(fin, fout, N)
print(f"Wrote {DST} ({DST.stat().st_size/1024:.1f} KB)")
