"""Convert BodyParts3D heart OBJs to the web JSON files in web/.

Usage:
  python tools/convert_heart_mesh.py <partof-heart.zip> <region-of-wall.zip> <out-dir>

Inputs (BodyParts3D 4.3i, CC BY-SA 2.1 JP, https://lifesciencedb.jp/bp3d/):
  partof-heart.zip : Heart FMA7088, PART-OF tree, representation BP48944
  region-of-wall.zip: Region of wall of heart FMA86212, IS-A tree, BP48592
                      (wall of ventricle + wall of left/right atrium)

Outputs: heart.json, lad.json, lcx.json, rca.json (positions + triangle indices,
normalised as described in the transform below). The source zips are not
vendored in git; the converted JSON files are, under the same CC BY-SA 2.1 JP
terms as the source mesh (see assets/ASSETS.md).
"""
import argparse, sys, zipfile, re, json, pathlib
parser = argparse.ArgumentParser()
parser.add_argument("partof_zip")
parser.add_argument("region_zip")
parser.add_argument("out_dir")
args = parser.parse_args()
partof = pathlib.Path(args.partof_zip)
region = pathlib.Path(args.region_zip)
LAD = {"74912","3860","3868","3872","3874","3876","3892"}
LCX = {"3895","74923"}
RCA = {"3802","3807","3815","3818","3837","3840","3847","3848"}
def fma_of(name):
    # filename contains FMA3_2_1 (version) and FMA7088 (heart) before the
    # component FMA, so take the LAST FMA code in the name.
    ms = re.findall(r"_FMA(\d+)_", name)
    return ms[-1] if ms else None
def parse_obj(text):
    verts=[]; faces=[]
    for line in text.splitlines():
        if line.startswith("v "):
            p=line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
        elif line.startswith("f "):
            idx=[]
            for tok in line.split()[1:]:
                idx.append(int(tok.split("/")[0])-1)
            assert len(idx)==3, f"non-triangle face: {line}"
            faces.append(tuple(idx))
    return verts, faces
# Collect raw groups first, to compute a shared bbox
raw = {"heart": [], "lad": [], "lcx": [], "rca": []}
# heart walls from region zip (only the 3 wall files)
with zipfile.ZipFile(region) as z:
    for n in z.namelist():
        if not n.endswith(".obj"): continue
        v,f = parse_obj(z.read(n).decode(errors="replace"))
        raw["heart"].append((n,v,f))
# vessels from partof zip
with zipfile.ZipFile(partof) as z:
    for n in z.namelist():
        if not n.endswith(".obj"): continue
        fma = fma_of(n)
        grp = None
        if fma in LAD: grp="lad"
        elif fma in LCX: grp="lcx"
        elif fma in RCA: grp="rca"
        if grp:
            v,f = parse_obj(z.read(n).decode(errors="replace"))
            raw[grp].append((n,v,f))
# global bbox
bmin=[1e18]*3; bmax=[-1e18]*3
for grp, items in raw.items():
    for _, verts, _ in items:
        for x,y,zz in verts:
            for i,val in enumerate((x,y,zz)):
                bmin[i]=min(bmin[i],val); bmax[i]=max(bmax[i],val)
size=[bmax[i]-bmin[i] for i in range(3)]
s=3.2/max(size)
c=[(bmin[i]+bmax[i])/2 for i in range(3)]
print("bbox min",bmin,"max",bmax,"size",size,"scale",s,"centre",c)
out={}
for grp, items in raw.items():
    pos=[]; idx=[]
    for _, verts, faces in items:
        off=len(pos)
        for x,y,zz in verts:
            # (bp_x, bp_y, bp_z) -> (three_x, three_y, three_z)
            pos.extend([round((x-c[0])*s,4), round((zz-c[2])*s,4), round(-(y-c[1])*s,4)])
        for a,b,cc in faces:
            idx.extend([a+off,b+off,cc+off])
    out[grp]={"positions":pos,"indices":idx,"vertices":len(pos)//3,"faces":len(idx)//3}
    print(grp, out[grp]["vertices"], "verts", out[grp]["faces"], "faces", len(items), "components")
out["_meta"]={
  "source":"BodyParts3D 4.3i, CC BY-SA 2.1 JP, https://lifesciencedb.jp/bp3d/",
  "heart":"Wall of ventricle FMA13884 + wall of left atrium FMA9531 + wall of right atrium FMA9457 (region of wall of heart, FMA86212, BP48592)",
  "vessels":"Coronary components from Heart FMA7088 PART-OF BP48944; trunk of left coronary artery FMA3855 excluded (shared by LAD/LCX)",
  "transform":"three_x=(bp_x-cx)*s, three_y=(bp_z-cz)*s, three_z=-(bp_y-cy)*s, s=3.2/max_bbox_dimension, centred at origin",
  "bbox_mm":{"min":bmin,"max":bmax},
  "note":"Educational reference model (adult male TARO), not patient-specific. BodyParts3D itself warns some parts were made/distorted by artists and may contain errors."
}
dest=pathlib.Path(args.out_dir)
dest.mkdir(parents=True, exist_ok=True)
for grp in ("heart","lad","lcx","rca"):
    (dest/f"{grp}.json").write_text(json.dumps(out[grp], separators=(",",":")))
    print(grp, (dest/f"{grp}.json").stat().st_size, "bytes")
(dest/"MANIFEST.json").write_text(json.dumps(out["_meta"], indent=2))
print("done")
