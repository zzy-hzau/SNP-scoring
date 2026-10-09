#!/usr/bin/env python3
"Build the chromosome-wise float32 conservation resources from raw WIG."
from phenofuncrank_repro.common.paths import project_root, package_dir
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = project_root(__file__)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",required=True,type=Path)
    parser.add_argument("--out-dir",required=True,type=Path)
    parser.add_argument("--profile",type=Path,default=ROOT/"configs/rice_uniform_conservation_20260926.json")
    parser.add_argument("--compiler",default="g++")
    parser.add_argument("--workers",default=4,type=int)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    args.out_dir.mkdir(parents=True,exist_ok=False)
    program=args.out_dir/"wig_to_dense"
    cpp=Path(__file__).with_name("conservation")/"wig_to_dense.cpp"
    subprocess.run([args.compiler,"-O2","-std=c++11",str(cpp),"-o",str(program)],check=True)
    def build(row):
        source=args.source_root/row["source_relative"]
        target=args.out_dir/row["output_filename"]
        subprocess.run([str(program),str(source),str(row["length"]),str(target)],check=True)
        if target.stat().st_size!=row["length"]*4:
            raise ValueError("Float32 track length mismatch")
        return {**row,"source_bytes":source.stat().st_size,
                "source_sha256":digest(source),"dense_bytes":target.stat().st_size,
                "dense_sha256":digest(target)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tracks=list(pool.map(build,profile["tracks"]))
    report={"profile":profile["profile"],"tracks":tracks,
            "compiler_source_sha256":digest(cpp),"dtype":"float32",
            "coordinate_convention":profile["coordinate_convention"],
            "missing_value":profile["missing_value"]}
    (args.out_dir/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
