"""Create a reader-safe Zurich H&E packet and an author-only spot coordinate key.

Supply a directory with KC1/2/3.h5ad, and full_resolution/{KC1,KC2,KC3}.tif
and *_manual_loupe_alignment.json from Zenodo 14620362. The output directory
must be new so a second run cannot silently change readers' slide identifiers.
"""
import argparse
import csv
import json
from pathlib import Path
from secrets import SystemRandom
import shutil
import tempfile
from zipfile import ZipFile, ZIP_STORED

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree
import tifffile

HERE = Path(__file__).resolve().parent


def resized(image, size, colour=True):
    src = Image.fromarray(image)
    if not colour:
        src = src.convert("L")
    return np.asarray(src.resize(size, Image.Resampling.BOX))


def make_slide(source, sample, case, dest):
    full = source / "full_resolution"
    with tifffile.TiffFile(full / f"{sample}.tif") as t:
        page = t.pages[0]
        if int(page.photometric) != 2 or int(page.tags.valueof("Orientation", 1)) != 1:
            raise ValueError(f"Check image orientation and colour encoding of {sample}")
        rgb = page.asarray()
        icc = page.tags.valueof(34675, None)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Unexpected TIFF shape or channel depth in {sample}")
    align = pd.DataFrame(json.loads((full / f"{sample}_manual_loupe_alignment.json").read_text())["oligo"])
    align = align.set_index(["row", "col"], verify_integrity=True)
    with h5py.File(source / f"{sample}.h5ad") as h:
        obs = h["obs"]
        ids = obs["_index"].asstr()[:]
        idx = pd.MultiIndex.from_arrays((obs["x_array"][:],obs["y_array"][:]), names=["row","col"])
        spots = align.loc[idx,["imageX","imageY"]].to_numpy()
        encoded = h["obsm/spatial"][:]
        hires = h["uns/spatial/library_id/images/hires"][:]
    # The archived h5ad hires image and coordinates are at one quarter of source size.
    height, width = rgb.shape[:2]
    if max(abs(height/4-hires.shape[0]),abs(width/4-hires.shape[1])) > 1:
        raise ValueError(f"Unexpected image scaling for {sample}")
    err = np.linalg.norm(spots-4*encoded,axis=1).max()
    corr = np.corrcoef(resized(rgb,(200,200),False).ravel(),resized(hires,(200,200),False).ravel())[0,1]
    if err > 1 or corr < .90 or not ((spots >= 0) & (spots < [width,height])).all():
        raise ValueError(f"Review original alignment for {sample}: max error {err:g}, image r {corr:g}")
    # Standard Visium has 100 µm centre spacing. Source Loupe alignments are in full-image pixels.
    pitch = np.median(cKDTree(spots).query(spots,k=2)[0][:,1])
    mpp = float(100/pitch)
    meta = {"axes":"YXS","Name":case,"PhysicalSizeX":mpp,"PhysicalSizeXUnit":"µm",
            "PhysicalSizeY":mpp,"PhysicalSizeYUnit":"µm"}
    path = dest/f"{case}.ome.tif"
    opts = dict(photometric="rgb", tile=(256,256), compression="deflate",compressionargs={"level":3},
                maxworkers=2, software=False)
    if icc is not None:
        opts["extratags"] = [(34675, "B",len(icc),icc, False)]
    with tifffile.TiffWriter(path,ome=True) as writer:
        writer.write(rgb,subifds=4,metadata=meta,resolution=(10000/mpp,10000/mpp),resolutionunit="CENTIMETER",**opts)
        for factor in (2,4,8,16):
            dimensions = (max(1,width//factor),max(1,height//factor))
            writer.write(resized(rgb,dimensions),subfiletype=1,metadata=None,**opts)
    with tifffile.TiffFile(path) as t:
        assert len(t.series[0].levels)==5 and np.array_equal(t.series[0].asarray(),rgb)
        if sample in (t.ome_metadata or ""):
            raise ValueError("Source identifier appeared in reader metadata")
    barcodes=pd.DataFrame({"case":case,"barcode":ids,"x_fullres":spots[:,0],"y_fullres":spots[:,1]})
    key={"case":case,"source_sample":sample,"width_px":width,"height_px":height,"um_per_px":mpp,
         "barcode_count":len(ids),"max_coordinate_error_px":err,"thumbnail_correlation":corr}
    return key,barcodes


def main(source,out):
    out=out.resolve()
    if out.exists():
        raise FileExistsError(f"Use a new output directory; an existing reader allocation is preserved at {out}")
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tls_reader_",dir=out.parent) as tmp:
        work=Path(tmp)/"packet";reader=work/"reader";private=work/"analyst_private"
        reader.mkdir(parents=True);private.mkdir(mode=0o700)
        samples=["KC1","KC2","KC3"];SystemRandom().shuffle(samples)
        keys=[];bc=[]
        for i,sample in enumerate(samples,1):
            key,spots=make_slide(source,sample,f"Slide_{i:02d}",reader)
            keys.append(key);bc.append(spots)
        pd.DataFrame(keys).to_csv(private/"case_key.csv",index=False)
        pd.concat(bc,ignore_index=True).to_csv(private/"spot_coordinates.csv.gz",index=False)
        shutil.copyfile(HERE/"tls_reader_instructions_zh.md",reader/"阅片说明.md")
        with (reader/"review_receipt.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.writer(f);w.writerow(["case","reader_code","complete_review","notes"])
            for k in keys:w.writerow([k["case"],"","",""])
        # The zip is what each reader receives; no source keys or algorithm outputs enter the archive.
        with ZipFile(work/"TLS_reader_packet.zip","w",compression=ZIP_STORED) as z:
            for path in sorted(reader.iterdir()):z.write(path,path.name)
        shutil.rmtree(reader)
        work.rename(out)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source",type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    main(args.source,args.out)
